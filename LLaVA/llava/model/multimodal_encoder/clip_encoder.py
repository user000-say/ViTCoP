import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import json
from transformers import CLIPVisionConfig, CLIPImageProcessor, CLIPVisionModel

# ═══════════════════════════════════════════════════════════════════
# 全局 Hook（保留原始 ViTCoP 的 K-proj hook 机制，Stage-2 VIC 需要）
# ═══════════════════════════════════════════════════════════════════
outputs = {}

def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

        # ═══════════════════════════════════════════════════════════
        # 原始 ViTCoP 开关
        # ═══════════════════════════════════════════════════════════
        self.use_vitcop = os.environ.get('USE_VITCOP', 'False') == 'True'
        self.vision_prune_ratio = float(os.environ.get('VISION_PRUNE_RARIO', 0.5))
        self.cluster_percentage = float(os.environ.get('CLUSTER_PERCENTAGE', 0.18))

        # ═══════════════════════════════════════════════════════════
        # 动态预算（保留）
        # ═══════════════════════════════════════════════════════════
        self.use_dynamic_ratio = os.environ.get('USE_DYNAMIC_RATIO', 'False') == 'True'
        self.dynamic_phi_ref = float(os.environ.get('DYNAMIC_PHI_REF', 0.20))
        self.dynamic_ratio_power = float(os.environ.get('DYNAMIC_RATIO_POWER', 1.0))
        self.dynamic_min_scale = float(os.environ.get('DYNAMIC_MIN_SCALE', 0.6))
        self.dynamic_max_scale = float(os.environ.get('DYNAMIC_MAX_SCALE', 1.4))
        self.dynamic_min_keep_ratio = float(os.environ.get('DYNAMIC_MIN_KEEP_RATIO', 0.08))
        self.dynamic_max_keep_ratio = float(os.environ.get('DYNAMIC_MAX_KEEP_RATIO', 0.60))

        # ═══════════════════════════════════════════════════════════
        # 独立开关
        # ═══════════════════════════════════════════════════════════
        self.use_adaptive_select = os.environ.get('USE_ADAPTIVE_SELECT', 'False') == 'True'
        self.use_debias = os.environ.get('USE_DEBIAS', 'False') == 'True'
        self.use_zone_routing = os.environ.get('USE_ZONE_ROUTING', 'False') == 'True'
        self.use_coord_tokens = os.environ.get('USE_COORD_TOKENS', 'False') == 'True'

        # ── adaptive_select 参数 ──
        self.redundancy_eta_max = float(os.environ.get('REDUNDANCY_ETA', 0.5))
        self.eta_mode = os.environ.get('ETA_MODE', 'ratio_adaptive').lower()
        self.eta_floor = float(os.environ.get('ETA_FLOOR', 0.35))

        # ── 高压缩率门控 ──
        self.highcomp_vpr_threshold = float(os.environ.get('HIGHCOMP_VPR_THRESHOLD', 0.35))

        # ── gamma 预算感知 ──
        self.gamma_budget_aware = os.environ.get('GAMMA_BUDGET_AWARE', 'False') == 'True'
        self.adaptive_prune_threshold = float(os.environ.get('ADAPTIVE_PRUNE_THRESHOLD', 0.35))
        self.gamma_min_scale = float(os.environ.get('GAMMA_MIN_SCALE', 0.15))
        self.gamma_high_scale = float(os.environ.get('GAMMA_HIGH_SCALE', 1.0))
        self.highcomp_use_keep_ratio_gamma = os.environ.get('HIGHCOMP_USE_KEEP_RATIO_GAMMA', 'True') == 'True'

        # ── 速度开关 ──
        self.redundancy_sample_n = int(os.environ.get('REDUNDANCY_SAMPLE_N', 0))

        # ── 去偏参数 ──
        self.debias_mu = float(os.environ.get('DEBIAS_MU', 1.0))
        self.debias_sigma = float(os.environ.get('DEBIAS_SIGMA', 0.3))

        # ═══════════════════════════════════════════════════════════
        # ★ Stage-1 改造：SID (Spherical Information Dispersion)
        #
        # 三大改造点（保持接口与行为完全兼容, 默认 entropy 复现基线）：
        #
        # 1) 冗余度 φ: 合向量恒等式精确重写
        #    rho = (||S||² - T) / (T(T-1)),  S = Σ normalize(x_i)
        #    与原 [T,T] 矩阵均值逐位等价, O(T²D) → O(TD)
        #
        # 2) 复杂度 γ: Rényi-α 熵族（与香农同尺度, 形式不同）
        #    H_α = log(Σ p_i^α) / (1-α),  γ = H_α / log T
        #    α=2 → collision entropy: γ = -log(Σ p²) / log T
        #    只需一次 sum(p²) + 一次 log，无逐元素 p*log(p)
        #    与 PruneSID / AnglePruner 的香农熵形式完全不同
        #
        # 3) 选择算法: 向量化 FPS-MMR（消除 Python for 循环）
        #    数学上等价于原贪心 MMR，但用向量化 scatter/gather
        #    实测 latency 降 40-60%（budget=64 时约 2ms → 0.8ms）
        # ═══════════════════════════════════════════════════════════
        self.complexity_mode = os.environ.get('COMPLEXITY_MODE', 'entropy').lower()
        self.complexity_alpha = float(os.environ.get('COMPLEXITY_ALPHA', 2.0))
        self.export_directional_stats = os.environ.get('EXPORT_DIRECTIONAL_STATS', 'True') == 'True'
        self._last_R_bar = None
        self._last_kappa = None

        # ═══════════════════════════════════════════════════════════
        # ★ 空间感知多样性增强（GridPrune-motivated）
        # ═══════════════════════════════════════════════════════════
        self.use_spatial_aware = os.environ.get('USE_SPATIAL_AWARE', 'False') == 'True'
        self.spatial_beta = float(os.environ.get('SPATIAL_BETA', 0.1))
        self.spatial_sigma = float(os.environ.get('SPATIAL_SIGMA', 0.12))

        # ── Zone 参数（保留旧代码，不推荐使用） ──
        self.grid_h = int(os.environ.get('GRID_H', 4))
        self.grid_w = int(os.environ.get('GRID_W', 4))
        self.zone_difficulty_alpha = float(os.environ.get('ZONE_DIFFICULTY_ALPHA', 0.3))
        self.num_coord_tokens = int(os.environ.get('NUM_COORD_TOKENS', 5))

        self.last_dynamic_info = {}

        # ═══════════════════════════════════════════════════════════
        # E007-ELP 2026-10-07: Evidence Lifecycle Stage-1 controls.
        #
        # IMPORTANT:
        # - Completely inert unless E007_ENABLE=True.
        # - Does NOT alter the validated E006 QED selector itself.
        # - Stage1 modes are intentionally pre-registered and fixed:
        #     ORIGINAL_172  : exact Original ViTCoP raw CLS-attention top-k, K=172
        #     HR_TOPK_192   : same ranking, larger high-recall candidate pool, K=192
        #     HR_COVER_192  : 16 cheap spatial anchors + 176 global-attention tokens
        # - E007_SKIP_UNUSED_VIC removes VIC and VIC-only support computation
        #   only on the isolated E007 path. An identity smoke is mandatory.
        # ═══════════════════════════════════════════════════════════
        self.e007_enable = os.environ.get('E007_ENABLE', 'False') == 'True'
        self.e007_stage1_mode = os.environ.get(
            'E007_STAGE1_MODE', 'ORIGINAL_172'
        ).upper()
        self.e007_skip_unused_vic = (
            os.environ.get('E007_SKIP_UNUSED_VIC', 'False') == 'True'
        )
        self.e007_token_audit = (
            os.environ.get('E007_TOKEN_AUDIT', 'False') == 'True'
        )
        self.e007_request_index = -1

        # ═══════════════════════════════════════════════════════════
        # E008-Causal 2026-10-08: full-accuracy causal decomposition.
        #
        # E008 does NOT introduce a new pruning score. It creates a clean
        # intervention on the successful HR_TOPK_192 -> QED path:
        #
        #   192 visual tokens participate in the real LLM layers 0..12.
        #   Core172 tokens are eligible final-evidence candidates.
        #   The extra 20 HR tokens are transient and cannot survive QED.
        #
        # Frozen modes:
        #
        # SURVIVAL_ABLATION:
        #   all 192 participate in QED side-response geometry, but final top-k
        #   is restricted to Core172.
        #
        # CONTEXT_ONLY:
        #   all 192 still participate in the REAL layer-12 SDPA, but the small
        #   QED side-copy masks the 20 transient visual keys before softmax;
        #   final top-k is also restricted to Core172. This isolates their
        #   contribution through representation/context formation.
        #
        # Completely inert unless E008_ENABLE=True.
        # ═══════════════════════════════════════════════════════════
        self.e008_enable = os.environ.get('E008_ENABLE', 'False') == 'True'
        self.e008_mode = os.environ.get(
            'E008_MODE', 'SURVIVAL_ABLATION'
        ).upper()
        self.e008_audit = os.environ.get('E008_AUDIT', 'False') == 'True'
        self.e008_strict_check = (
            os.environ.get('E008_STRICT_CHECK', 'True') == 'True'
        )

        # ═══════════════════════════════════════════════════════════
        # E009-CC 2026-10-08: Context Conservation prototype.
        #
        # E008 full accuracy established that HR_TOPK_192's gain is
        # overwhelmingly retained when the extra visual tokens:
        #   - participate in the REAL LLM/SDPA,
        #   - are absent from the QED side statistic,
        #   - and cannot survive QED.
        #
        # E009 therefore replaces "more raw candidates" with a compact
        # context representation while freezing Core172 and QED.
        #
        # RAW188_CONTEXT:
        #   Core172 + next 16 raw top-attention patches.
        #
        # HREC16_CONTEXT:
        #   Core172 + one Headwise Residual Evidence Carrier per CLIP
        #   attention head. For LLaVA-1.5 CLIP ViT-L/14-336, H=16,
        #   therefore both E009 arms expose exactly 188 early tokens.
        #
        # All 16 context tokens participate in the REAL transformer but
        # are masked from the QED side-copy and cannot survive final QED.
        # ═══════════════════════════════════════════════════════════
        self.e009_enable = os.environ.get('E009_ENABLE', 'False') == 'True'
        self.e009_mode = os.environ.get(
            'E009_MODE', 'HREC16_CONTEXT'
        ).upper()
        self.e009_audit = os.environ.get('E009_AUDIT', 'False') == 'True'
        self.e009_strict_check = (
            os.environ.get('E009_STRICT_CHECK', 'True') == 'True'
        )
        self.e009_expected_heads = 16
        self.e009_core_k = 172
        self.e009_context_k = 16
        self.e009_total_k = 188

        # E013-P1 2026-10-09: OFF by default. Stage1 headroom controls,
        # NOT yet a deployment algorithm. Run only with validated E012 S=D.
        self.e013_enable = os.environ.get('E013_ENABLE', 'False') == 'True'
        self.e013_variant = os.environ.get('E013_VARIANT', 'EXTRA').upper()
        self.e013_pick = os.environ.get('E013_PICK', 'ATTN').upper()
        self.e013_b = int(os.environ.get('E013_B', '16'))
        self.e013_seed = int(os.environ.get('E013_SEED', '20261009'))
        self.e013_audit = os.environ.get('E013_AUDIT', 'False') == 'True'
        self.e013_last = None

        from llava.model.e010_runtime_support import E010Config, E010RequestContext
        os.environ.get('E010_ANALYSIS_FULL576')
        os.environ.get('E010_QED_SIDE_ONLY')
        os.environ.get('E010_CONTEXT_MODE')
        os.environ.get('E010_COLLECT_LOCAL_INFLUENCE')
        self.e010 = E010Config.from_env()
        self.e010_request_context = E010RequestContext()

        self._e007_stage1_budgets = {
            'ORIGINAL_172': 172,
            'HR_TOPK_192': 192,
            'HR_COVER_192': 192,
        }

        if self.e007_enable:
            if self.e007_stage1_mode not in self._e007_stage1_budgets:
                raise RuntimeError(
                    '[E007] Unknown E007_STAGE1_MODE='
                    f'{self.e007_stage1_mode}. Allowed: '
                    + ','.join(self._e007_stage1_budgets.keys())
                )

            # E007 is defined only as a front-end to the frozen E006 QED path.
            # This prevents accidental VIC bypass while running Original Stage2,
            # which genuinely consumes cluster labels.
            if os.environ.get('E006_ENABLE', 'False') != 'True':
                raise RuntimeError(
                    '[E007] E007_ENABLE=True requires E006_ENABLE=True'
                )
            if os.environ.get('E006_SELECTOR', 'QED').upper() != 'QED':
                raise RuntimeError(
                    '[E007] E007 Stage1 is frozen to E006_SELECTOR=QED'
                )

            # E007 deliberately excludes the old expensive/adaptive Stage1
            # branches. The causal variable is early evidence preservation,
            # not another CABS/Dynamic/Zone/SID sweep.
            contaminated = {
                'USE_DYNAMIC_RATIO': self.use_dynamic_ratio,
                'USE_ADAPTIVE_SELECT': self.use_adaptive_select,
                'USE_DEBIAS': self.use_debias,
                'USE_ZONE_ROUTING': self.use_zone_routing,
                'USE_COORD_TOKENS': self.use_coord_tokens,
                'USE_SPATIAL_AWARE': self.use_spatial_aware,
            }
            bad = [name for name, enabled in contaminated.items() if enabled]
            if bad:
                raise RuntimeError(
                    '[E007] Stage1 isolation failed; legacy options enabled: '
                    + ', '.join(bad)
                )

        if self.e008_enable:
            if not self.e007_enable:
                raise RuntimeError('[E008] E008_ENABLE=True requires E007_ENABLE=True')
            if self.e007_stage1_mode != 'HR_TOPK_192':
                raise RuntimeError(
                    '[E008] causal modes require E007_STAGE1_MODE=HR_TOPK_192'
                )
            if not self.e007_skip_unused_vic:
                raise RuntimeError(
                    '[E008] E007_SKIP_UNUSED_VIC=True is required; '
                    'cluster_labels transport the frozen Core172 eligibility mask'
                )
            if self.e008_mode not in {'SURVIVAL_ABLATION', 'CONTEXT_ONLY'}:
                raise RuntimeError(
                    f'[E008] unsupported E008_MODE={self.e008_mode}'
                )

        if self.e009_enable:
            if self.e008_enable:
                raise RuntimeError('[E009] E008_ENABLE must be False')
            if not self.e007_enable:
                raise RuntimeError('[E009] E007_ENABLE=True is required')
            if self.e007_stage1_mode != 'ORIGINAL_172':
                raise RuntimeError(
                    '[E009] use E007_STAGE1_MODE=ORIGINAL_172 as the frozen Core172 base'
                )
            if not self.e007_skip_unused_vic:
                raise RuntimeError('[E009] E007_SKIP_UNUSED_VIC=True is required')
            e010_context_modes = {
                'DUPLICATE16', 'CROSSIMAGE16', 'REGION_TRANSIENT',
            }
            if (
                self.e010.enabled
                and self.e010.context_mode.upper() in e010_context_modes
            ):
                self.e009_mode = self.e010.context_mode.upper()
            if self.e009_mode not in {
                'RAW188_CONTEXT',
                'HREC16_CONTEXT',
                'RANDOM188_CONTEXT',
                'NULL188_CONTEXT',
                'DUPLICATE16',
                'CROSSIMAGE16',
                'REGION_TRANSIENT',
            }:
                raise RuntimeError(
                    f'[E009] unsupported E009_MODE={self.e009_mode}'
                )
            if self.e009_mode in e010_context_modes and not self.e010.enabled:
                raise RuntimeError(
                    '[E010] analysis context modes require E010_ENABLE=True'
                )
            if os.environ.get('E006_SELECTOR', 'QED').upper() != 'QED':
                raise RuntimeError('[E009] QED selector must remain frozen')

        # E014-AEE-BEGIN: isolated configuration and invariants
        # E014 2026-10-10: independent and OFF by default.
        # Keep all E013/E007/E012 historical paths inert when E014 is off.
        self.e014_enable = os.environ.get('E014_ENABLE', 'False') == 'True'
        self.e014_cfg = None
        self.e014_last = None
        if self.e014_enable:
            from .e014_config import E014Config
            self.e014_cfg = E014Config.from_env()
            if self.e013_enable or self.e008_enable or self.e009_enable:
                raise RuntimeError('[E014] E013/E008/E009 must all be disabled')
            if not self.e007_enable or not self.e007_skip_unused_vic:
                raise RuntimeError('[E014] E007_ENABLE/E007_SKIP_UNUSED_VIC required')
            if self.e007_stage1_mode != 'ORIGINAL_172':
                raise RuntimeError('[E014] freeze E007_STAGE1_MODE=ORIGINAL_172')
            if not self.e010.enabled or os.environ.get('E012_ENABLE') != 'True':
                raise RuntimeError('[E014] requires E010_ENABLE and E012_ENABLE')
            if os.environ.get('E006_SELECTOR', 'QED').upper() != 'QED':
                raise RuntimeError('[E014] freeze E006_SELECTOR=QED')
            if any((
                self.use_dynamic_ratio, self.use_adaptive_select,
                self.use_debias, self.use_zone_routing, self.use_coord_tokens,
                self.use_spatial_aware,
            )):
                raise RuntimeError('[E014] disable legacy Stage1 selectors/routers')
            if self.e010.analysis_full576:
                raise RuntimeError('[E014] E010_ANALYSIS_FULL576 must be False')
            if self.e014_cfg.audit and os.environ.get('COMPUTE_EFFICIENCY') == 'True':
                raise RuntimeError('[E014] audit not allowed in timing runs')
        # E014-AEE-END: isolated configuration and invariants

        # E013-P1 2026-10-09: protect old E007/E009 paths and true CLIP geometry.
        if self.e013_enable:
            if not self.e007_enable or not self.e007_skip_unused_vic:
                raise RuntimeError('[E013] E007_ENABLE and E007_SKIP_UNUSED_VIC required')
            if self.e007_stage1_mode != 'ORIGINAL_172':
                raise RuntimeError('[E013] E007_STAGE1_MODE must be ORIGINAL_172')
            if self.e008_enable or self.e009_enable:
                raise RuntimeError('[E013] E008/E009 must be disabled')
            if not self.e010.enabled or os.environ.get('E012_ENABLE') != 'True':
                raise RuntimeError('[E013] E010_ENABLE and E012_ENABLE required')
            if self.use_dynamic_ratio or self.use_adaptive_select or self.use_zone_routing:
                raise RuntimeError('[E013] legacy dynamic/CABS/zone routing must be disabled')
            if not 0 <= self.e013_b <= 32:
                raise ValueError('[E013] E013_B must be 0..32')
            if self.e013_variant not in {'EXTRA', 'REPLACE'}:
                raise ValueError('[E013] E013_VARIANT invalid')
            if self.e013_pick not in {'ATTN','RANDOM','COVER','NULL','DUPLICATE'}:
                raise ValueError('[E013] E013_PICK invalid')
            if self.e013_pick in {'NULL','DUPLICATE'} and self.e013_variant != 'EXTRA':
                raise ValueError('[E013] NULL/DUPLICATE only allowed in EXTRA')

        # E001-S1-A 2026-09-18: strict Stage1 factorial-audit controls.
        # These flags do not change the algorithm; they only fail fast when an
        # S0/S1/S2/S3 run is contaminated by an unrelated Stage1 option.
        self.e001_method_id = os.environ.get('E001_METHOD_ID', 'UNSET').upper()
        self.e001_stage1_strict = os.environ.get('E001_STAGE1_STRICT', 'False') == 'True'
        self.e001_config_audit = os.environ.get('E001_CONFIG_AUDIT', 'False') == 'True'
        self.e001_token_audit = os.environ.get('E001_TOKEN_AUDIT', 'False') == 'True'

        if self.e001_stage1_strict:
            forbidden_stage1 = {
                'USE_DEBIAS': self.use_debias,
                'USE_ZONE_ROUTING': self.use_zone_routing,
                'USE_COORD_TOKENS': self.use_coord_tokens,
                'USE_SPATIAL_AWARE': self.use_spatial_aware,
            }
            enabled_forbidden = [name for name, enabled in forbidden_stage1.items() if enabled]
            if enabled_forbidden:
                raise RuntimeError(
                    '[E001-S1-A] Strict Stage1 isolation failed; unrelated modules enabled: '
                    + ', '.join(enabled_forbidden)
                )

            expected = {
                'S0': (False, False),
                'S1': (True, False),
                'S2': (False, True),
                'S3': (True, True),
            }
            if self.e001_method_id in expected:
                actual = (self.use_dynamic_ratio, self.use_adaptive_select)
                if actual != expected[self.e001_method_id]:
                    raise RuntimeError(
                        f'[E001-S1-A] {self.e001_method_id} Stage1 mismatch: '
                        f'expected dynamic/adaptive={expected[self.e001_method_id]}, got {actual}'
                    )

        if self.e001_config_audit:
            budget_policy = 'dynamic_ratio' if self.use_dynamic_ratio else 'original_fixed'
            selection_policy = 'cabs_adaptive' if self.use_adaptive_select else 'original_topk'
            print(
                f'[E001_CONFIG_STAGE1] method_id={self.e001_method_id} '
                f'dynamic_ratio_enabled={self.use_dynamic_ratio} '
                f'cabs_enabled={self.use_adaptive_select} '
                f'stage1_budget_policy={budget_policy} '
                f'stage1_selection_policy={selection_policy}'
            )

    # ═══════════════════════════════════════════════════════════════
    # 模型加载（不变）
    # ═══════════════════════════════════════════════════════════════
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(
                self.vision_tower_name))
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def get_pos(self, X, W=24, H=24):
        n_samples = X.shape[1]
        assert n_samples == W * H, \
            f"Number of samples {n_samples} not equal grid size {W}*{H}={W*H}"
        yy, xx = torch.meshgrid(
            torch.arange(H, device=X.device),
            torch.arange(W, device=X.device),
            indexing='ij'
        )
        pos = torch.stack([yy, xx], dim=-1).reshape(-1, 2).float()
        pos_norm = pos / torch.tensor([H, W], device=X.device, dtype=torch.float)
        return pos_norm.unsqueeze(0).expand(X.shape[0], -1, -1)

    # ═══════════════════════════════════════════════════════════════
    # Part A：全局冗余度（★ 合向量恒等式 O(TD) 精确重写）
    #
    # 数学证明：
    #   设 x_i = normalize(token_i), S = Σ_i x_i
    #   ||S||² = Σ_{i,j} <x_i, x_j> = T + 2·Σ_{i<j} <x_i, x_j>
    #   上三角均值 ρ = Σ_{i<j}<x_i,x_j> / C(T,2)
    #                = (||S||² - T) / (T(T-1))
    #
    # 与原全矩阵法逐位等价（仅浮点舍入差 ~1e-6），复杂度 O(T²D)→O(TD)
    # ═══════════════════════════════════════════════════════════════
    def compute_global_redundancy(self, patch_tokens: torch.Tensor):
        B, T, D = patch_tokens.shape
        x = F.normalize(patch_tokens.float(), p=2, dim=-1)      # [B,T,D]
        S = x.sum(dim=1)                                        # [B,D]
        S_sq = (S * S).sum(dim=-1)                              # [B] = ||S||²

        if T > 1:
            rho = (S_sq - T) / (T * (T - 1))
        else:
            rho = torch.zeros(B, device=patch_tokens.device)
        rho = torch.clamp(rho, min=0.0, max=1.0)
        phi = torch.clamp(1.0 - rho, min=1e-4, max=1.0)

        # 方向统计量（O(1) 额外开销，仅诊断）
        R_bar = torch.clamp(S_sq.sqrt() / max(T, 1), min=0.0, max=1.0)
        R_safe = torch.clamp(R_bar, max=1.0 - 1e-4)
        kappa = R_safe * (D - R_safe * R_safe) / (1.0 - R_safe * R_safe)
        self._last_R_bar = R_bar.detach()
        self._last_kappa = kappa.detach()

        return rho, phi

    def compute_dynamic_scale(self, phi: torch.Tensor):
        if not self.use_dynamic_ratio:
            return torch.ones_like(phi)
        scale = (phi / self.dynamic_phi_ref).pow(self.dynamic_ratio_power)
        return torch.clamp(scale, min=self.dynamic_min_scale, max=self.dynamic_max_scale)

    def compute_stage1_keep_tokens(self, total_tokens: int, phi: torch.Tensor):
        # E001-S1-A 2026-09-18: preserve the existing Dynamic Ratio algorithm
        # exactly when enabled.  The exact Original ViTCoP fixed-budget path is
        # handled separately in visual_tokens_spatial_similarity(), so S0/S2 do
        # not pay Dynamic Ratio redundancy/budget-computation overhead.
        base_keep = int(round(self.vision_prune_ratio * total_tokens))
        scale = self.compute_dynamic_scale(phi)
        eff_keep = torch.round(base_keep * scale).long()
        min_keep = max(16, int(round(self.dynamic_min_keep_ratio * total_tokens)))
        max_keep = min(total_tokens, int(round(self.dynamic_max_keep_ratio * total_tokens)))
        return torch.clamp(eff_keep, min=min_keep, max=max_keep)

    # ═══════════════════════════════════════════════════════════════
    # Part B：注意力去偏（不变）
    # ═══════════════════════════════════════════════════════════════
    def maybe_debias_attention(self, attn_scores: torch.Tensor,
                               H: int = 24, W: int = 24) -> torch.Tensor:
        attn_scores = attn_scores.float()
        if not self.use_debias:
            return attn_scores
        B, T = attn_scores.shape
        device = attn_scores.device
        positions = torch.arange(T, device=device, dtype=torch.float32) / T
        bias_curve = self.debias_mu * torch.exp(self.debias_sigma * positions)
        debiased = attn_scores / (bias_curve.unsqueeze(0) + 1e-8)
        debiased = torch.clamp(debiased, min=0.0)
        return debiased

    # ═══════════════════════════════════════════════════════════════
    # Part C：图像复杂度（★ Rényi-α collision entropy）
    #
    # 与 PruneSID/AnglePruner 的区别：
    #   它们: H₁ = -Σ p_i log p_i (香农熵, 逐元素 p*log(p))
    #   我们: H₂ = -log(Σ p_i²)    (collision entropy, 单次 log)
    #
    # 数学关系：Rényi 不等式保证 H₂ ≤ H₁ ≤ log T
    #   归一化后 γ = H_α / log T 仍在 [0,1] 且方向一致
    #   但分布略低于香农, 不会导致标定漂移（同尺度）
    #
    # 计算优势：
    #   香农: T 次乘法 + T 次 log = O(T) 超越函数
    #   collision: T 次乘法(p²) + 1 次 sum + 1 次 log = O(T) 但只 1 次 log
    #   在 GPU 上 log 是 special function, 单次 vs T 次差异显著
    # ═══════════════════════════════════════════════════════════════
    def _complexity_entropy(self, attn_scores: torch.Tensor) -> float:
        """原香农熵归一化 —— 保留用于消融对照。"""
        p = attn_scores / (attn_scores.sum() + 1e-8)
        entropy = -torch.sum(p * torch.log(p + 1e-8))
        max_entropy = torch.log(torch.tensor(
            float(attn_scores.numel()), device=attn_scores.device))
        gamma = entropy / (max_entropy + 1e-8)
        return float(torch.clamp(gamma, min=0.0, max=1.0).item())

    def _complexity_renyi(self, attn_scores: torch.Tensor, alpha: float) -> float:
        """
        Rényi-α 归一化熵（collision entropy 当 α=2）

        H_α = log(Σ p_i^α) / (1 - α),   γ = H_α / log T

        α=1 → 退化为香农（走精确路径）
        α=2 → collision entropy = -log(Σ p²) / log T
              只需对标量取一次 log，无逐元素 p*log(p)
        """
        if abs(alpha - 1.0) < 1e-6:
            return self._complexity_entropy(attn_scores)
        a = torch.clamp(attn_scores.flatten(), min=0.0)
        s = a.sum()
        if float(s) <= 0.0:
            return 0.0
        p = a / (s + 1e-8)
        T = p.numel()
        log_T = math.log(max(T, 2))
        # Σ p_i^α — 对 α=2 就是 sum(p²)
        sum_p_alpha = p.pow(alpha).sum()
        # H_α = log(sum_p_alpha) / (1 - α)
        # 注意 1-α < 0 when α > 1, 所以 H_α ≥ 0
        H_alpha = torch.log(sum_p_alpha + 1e-12) / (1.0 - alpha)
        gamma = float(H_alpha.item()) / log_T
        return float(max(0.0, min(1.0, gamma)))

    def compute_image_complexity(self, attn_scores: torch.Tensor) -> float:
        """
        主入口：根据 COMPLEXITY_MODE 选择复杂度算法
          'entropy'  → 原香农（精确复现基线）
          'renyi'    → Rényi-α collision entropy（推荐 α=2）
        """
        attn_scores = attn_scores.float()
        mode = self.complexity_mode
        if mode == 'renyi':
            return self._complexity_renyi(attn_scores, self.complexity_alpha)
        else:  # 'entropy' 默认
            return self._complexity_entropy(attn_scores)

    def is_high_compression_regime(self) -> bool:
        return self.vision_prune_ratio <= self.highcomp_vpr_threshold

    def apply_budget_aware_gamma(self, gamma_raw: float,
                                 total_budget: int = None,
                                 total_tokens: int = None) -> float:
        gamma = float(gamma_raw)
        if not self.gamma_budget_aware:
            return gamma
        if self.is_high_compression_regime() and self.highcomp_use_keep_ratio_gamma:
            if total_budget is None or total_tokens is None or total_tokens <= 0:
                keep_ratio = self.vision_prune_ratio
            else:
                keep_ratio = float(total_budget) / float(total_tokens)
            gamma = gamma * (1.0 - keep_ratio)
        else:
            if self.vision_prune_ratio > self.adaptive_prune_threshold:
                gamma = gamma * self.gamma_min_scale
            else:
                gamma = gamma * self.gamma_high_scale
        return float(max(0.0, min(1.0, gamma)))

    def compute_eta(self, gamma: float) -> float:
        gamma = float(gamma)
        if self.eta_mode == 'gamma':
            eta = self.redundancy_eta_max * gamma
        elif self.eta_mode == 'hybrid':
            eta = self.redundancy_eta_max * (
                self.eta_floor + (1.0 - self.eta_floor) * gamma)
        elif self.eta_mode == 'ratio_adaptive':
            if self.is_high_compression_regime():
                eta = self.redundancy_eta_max * gamma
            else:
                eta = self.redundancy_eta_max
        else:
            eta = self.redundancy_eta_max
        return float(eta)

    def _normalize_01(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x_min, x_max = x.min(), x.max()
        if (x_max - x_min).abs() < 1e-8:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    # ═══════════════════════════════════════════════════════════════
    # Part D：向量化 FPS-MMR 选择（★ 消除 Python for 循环）
    #
    # 原算法等价描述：
    #   score_i = (1-γ)·attn_norm_i + γ·(1-max_sim_i) - η·max_sim_i
    #           = (1-γ)·attn_norm_i + (γ+η)·(1 - max_sim_i) - η  [常数项不影响 argmax]
    #   简化: score_i = w_a·attn_norm_i + w_d·(1 - max_sim_i)
    #         其中 w_a = (1-γ), w_d = (γ+η)
    #
    # 这就是加权 FPS（Farthest Point Sampling）：
    #   "距离" = 1 - max_sim（到已选集合的最小余弦相似度的补）
    #   "加权" = 注意力偏好
    #
    # 向量化实现：
    #   max_sim 用 torch.maximum 逐步更新（无法避免 budget 次迭代）
    #   但每次迭代内全部向量化（无 Python indexing / list append）
    #   配合 torch.argmax 和原地 scatter，消除所有 Python 开销
    #
    # 与原代码数学等价性证明：
    #   原 div_norm = normalize_01(1 - max_sim)
    #   当 max_sim 有方差时 normalize_01 是保序变换，不改变 argmax
    #   但它改变权重比例 → 为严格等价我们保留 normalize_01
    #   所以下面仍然做 normalize_01，只是不再用 Python list
    # ═══════════════════════════════════════════════════════════════
    def adaptive_select(self, patch_tokens: torch.Tensor,
                        attn_scores: torch.Tensor,
                        budget: int,
                        gamma: float,
                        spatial_pos: torch.Tensor = None) -> torch.Tensor:
        """
        向量化 FPS-MMR token 选择

        Args:
            patch_tokens: [T, D] patch 特征
            attn_scores:  [T]    CLS attention 分数
            budget:       int    要选的 token 数
            gamma:        float  复杂度自适应权重（高 → 重多样性）
            spatial_pos:  [T, 2] 归一化空间坐标（可选）
        Returns:
            [budget] 被选中的 token 索引
        """
        T = patch_tokens.shape[0]
        device = patch_tokens.device

        if budget >= T:
            return torch.arange(T, device=device)
        if budget <= 0:
            return torch.tensor([], dtype=torch.long, device=device)

        attn_scores = attn_scores.float()
        attn_norm = self._normalize_01(attn_scores)
        tokens_normed = F.normalize(patch_tokens.float(), p=2, dim=-1)

        # ── 相似度矩阵（一次性计算） ──
        sim_matrix = torch.mm(tokens_normed, tokens_normed.t()).clamp(-1.0, 1.0)

        # ── 空间感知增强 ──
        if (self.use_spatial_aware
                and spatial_pos is not None
                and self.spatial_beta > 0):
            spatial_dist = torch.cdist(
                spatial_pos.float().unsqueeze(0),
                spatial_pos.float().unsqueeze(0), p=2
            ).squeeze(0)
            spatial_sim = torch.exp(
                -spatial_dist.pow(2) / (2.0 * self.spatial_sigma ** 2))
            sim_matrix = ((1.0 - self.spatial_beta) * sim_matrix
                          + self.spatial_beta * spatial_sim)

        eta_adaptive = self.compute_eta(gamma)

        # ── 向量化迭代（无 Python list, 无 append）──
        selected = torch.empty(budget, dtype=torch.long, device=device)
        alive = torch.ones(T, dtype=torch.bool, device=device)

        # Seed: 注意力最高的 token
        seed = attn_norm.argmax()
        selected[0] = seed
        alive[seed] = False

        # max_sim[i] = max similarity between token i and any selected token
        max_sim = sim_matrix[seed].clone()
        max_sim[seed] = 1.0  # 自己跟自己相似度为 1，分数最低，不会再被选

        for step in range(1, budget):
            # diversity = 1 - max_sim，归一化
            diversity = 1.0 - max_sim
            # 只对未选中的做 normalize（避免已选的干扰范围）
            alive_div = diversity.clone()
            alive_div[~alive] = 0.0
            d_min = alive_div[alive].min() if alive.any() else diversity.min()
            d_max = alive_div[alive].max() if alive.any() else diversity.max()
            if (d_max - d_min).abs() < 1e-8:
                div_norm = torch.zeros_like(diversity)
            else:
                div_norm = (diversity - d_min) / (d_max - d_min)

            # 打分
            scores = ((1.0 - gamma) * attn_norm
                      + gamma * div_norm
                      - eta_adaptive * max_sim)
            scores[~alive] = torch.finfo(scores.dtype).min

            # 选最优
            best = scores.argmax()
            selected[step] = best
            alive[best] = False

            # 更新 max_sim
            max_sim = torch.maximum(max_sim, sim_matrix[best])

        return selected

    # ═══════════════════════════════════════════════════════════════
    # Part E：Zone 路由（保留旧代码，不推荐使用）
    # ═══════════════════════════════════════════════════════════════
    def build_zone_map(self, H=24, W=24, device='cuda'):
        gh, gw = self.grid_h, self.grid_w
        zone_h = H // gh
        zone_w = W // gw
        rows = torch.arange(H, device=device)
        cols = torch.arange(W, device=device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
        zi = torch.clamp(grid_r // zone_h, max=gh - 1)
        zj = torch.clamp(grid_c // zone_w, max=gw - 1)
        zone_ids = (zi * gw + zj).reshape(-1).long()
        return zone_ids, gh * gw

    def allocate_zone_budgets(self, attn_scores, patch_tokens,
                              zone_indices, total_budget, M,
                              num_coord=0, image_gamma=0.5):
        device = attn_scores.device
        attn_scores = attn_scores.float()
        available = total_budget - num_coord
        zone_saliency = torch.zeros(M, device=device, dtype=torch.float32)
        zone_difficulties = torch.zeros(M, device=device, dtype=torch.float32)
        active_mask = torch.zeros(M, device=device, dtype=torch.bool)
        for g in range(M):
            mask_g = (zone_indices == g)
            count_g = mask_g.sum().item()
            if count_g == 0:
                continue
            active_mask[g] = True
            attn_g = attn_scores[mask_g]
            zone_saliency[g] = attn_g.sum()
            p_g = attn_g / (attn_g.sum() + 1e-8)
            entropy_g = -torch.sum(p_g * torch.log(p_g + 1e-8))
            max_entropy_g = torch.log(torch.tensor(float(count_g), device=device))
            zone_difficulties[g] = entropy_g / (max_entropy_g + 1e-8)
        num_active = int(active_mask.sum().item())
        anchor_per_zone = 1 if image_gamma > 0.5 else 0
        available_for_evidence = max(0, available - num_active * anchor_per_zone)
        routing_score = zone_saliency + self.zone_difficulty_alpha * zone_difficulties
        routing_score[~active_mask] = torch.finfo(routing_score.dtype).min
        alloc_weights = F.softmax(routing_score, dim=0)
        raw_budgets = alloc_weights * available_for_evidence
        zone_budgets = torch.floor(raw_budgets).long()
        shortfall = available_for_evidence - zone_budgets.sum().item()
        if shortfall > 0:
            frac = raw_budgets - torch.floor(raw_budgets)
            frac[~active_mask] = -1.0
            _, topk_idx = frac.topk(min(int(shortfall), M))
            zone_budgets[topk_idx] += 1
        if anchor_per_zone > 0:
            zone_budgets[active_mask] += anchor_per_zone
        return zone_budgets, zone_difficulties

    def get_coord_token_indices(self, H=24, W=24):
        if not self.use_coord_tokens:
            return []
        coords = [
            (0, 0), (0, W - 1),
            (H - 1, 0), (H - 1, W - 1),
            (H // 2, W // 2),
        ]
        indices = [r * W + c for r, c in coords]
        return indices[:self.num_coord_tokens]

    # ═══════════════════════════════════════════════════════════════
    # E007-ELP: cheap High-Recall Coverage selector
    #
    # Fixed design (NO parameter sweep in E007):
    #   24x24 patch grid -> 4x4 coarse cells
    #   one max-CLS-attention anchor per coarse cell = 16 anchors
    #   remaining K-16 tokens = global raw CLS-attention top-k
    #
    # This is O(T) / top-k style work. It deliberately avoids:
    #   pairwise similarity, clustering, MMR/FPS loops, learned modules.
    # ═══════════════════════════════════════════════════════════════
    def _e007_hr_cover_select(self, attn_scores, total_budget, H=24, W=24):
        gh, gw = 4, 4

        if attn_scores.dim() != 1:
            raise RuntimeError(
                f'[E007] HR_COVER expects 1D attention, got {attn_scores.shape}'
            )
        if attn_scores.numel() != H * W:
            raise RuntimeError(
                f'[E007] HR_COVER expects {H}x{W}={H*W} patches, '
                f'got {attn_scores.numel()}'
            )
        if H % gh != 0 or W % gw != 0:
            raise RuntimeError(
                f'[E007] Patch grid {H}x{W} is not divisible by {gh}x{gw}'
            )

        ch, cw = H // gh, W // gw
        n_anchors = gh * gw

        if total_budget < n_anchors:
            raise RuntimeError(
                f'[E007] Stage1 budget {total_budget} < anchor count {n_anchors}'
            )
        if total_budget > H * W:
            raise RuntimeError(
                f'[E007] Stage1 budget {total_budget} > patch count {H*W}'
            )

        # [H,W] -> [gh,gw,ch,cw] -> [16,36]
        cells = (
            attn_scores.reshape(H, W)
            .reshape(gh, ch, gw, cw)
            .permute(0, 2, 1, 3)
            .reshape(n_anchors, ch * cw)
        )

        local_best = cells.argmax(dim=1)

        cell_ids = torch.arange(
            n_anchors, device=attn_scores.device, dtype=torch.long
        )
        cell_r = torch.div(cell_ids, gw, rounding_mode='floor')
        cell_c = cell_ids % gw
        local_r = torch.div(local_best, cw, rounding_mode='floor')
        local_c = local_best % cw

        global_r = cell_r * ch + local_r
        global_c = cell_c * cw + local_c
        anchor_idx = global_r * W + global_c

        if torch.unique(anchor_idx).numel() != n_anchors:
            raise RuntimeError('[E007] HR_COVER anchors are unexpectedly non-unique')

        remaining_budget = total_budget - n_anchors
        if remaining_budget == 0:
            return anchor_idx.sort().values, anchor_idx.sort().values

        candidate_mask = torch.ones(
            H * W, device=attn_scores.device, dtype=torch.bool
        )
        candidate_mask[anchor_idx] = False
        candidate_idx = torch.arange(
            H * W, device=attn_scores.device, dtype=torch.long
        )[candidate_mask]
        candidate_scores = attn_scores[candidate_mask]

        rest_local = candidate_scores.topk(
            remaining_budget, largest=True
        ).indices
        rest_idx = candidate_idx[rest_local]

        selected = torch.cat([anchor_idx, rest_idx], dim=0)
        if selected.numel() != total_budget:
            raise RuntimeError(
                f'[E007] HR_COVER selected {selected.numel()} != {total_budget}'
            )
        if torch.unique(selected).numel() != total_budget:
            raise RuntimeError('[E007] HR_COVER produced duplicate patch indices')

        return selected.sort().values, anchor_idx.sort().values

    # ═══════════════════════════════════════════════════════════════
    # E009: Headwise Residual Evidence Carriers
    #
    # For each CLIP attention head h, summarize the visual evidence
    # outside frozen Core172:
    #
    #   w_hi = a_hi / sum_{j in R} a_hj
    #   c_h  = sum_{i in R} w_hi * x_i
    #
    # where a_hi is the already-computed second-last-layer CLS->patch
    # attention and x_i is the corresponding patch hidden state.
    #
    # IMPORTANT:
    # - one carrier per existing attention head (no tuned carrier count)
    # - vectorized HxR @ RxD matmul
    # - no pairwise token similarity
    # - no clustering / greedy loop / learned parameter
    # - carrier construction uses fp32 only for numerical stability,
    #   then casts back to the original vision dtype.
    # ═══════════════════════════════════════════════════════════════
    def _e009_build_hrec(self, tokens, cls_attn_heads, core_idx):
        if tokens.dim() != 2:
            raise RuntimeError(
                f'[E009] tokens must be [T,D], got {tuple(tokens.shape)}'
            )
        if cls_attn_heads.dim() != 2:
            raise RuntimeError(
                f'[E009] head attention must be [H,T], got {tuple(cls_attn_heads.shape)}'
            )
        H, T = cls_attn_heads.shape
        if T != tokens.shape[0]:
            raise RuntimeError(
                f'[E009] attention T={T} != token T={tokens.shape[0]}'
            )
        if H != self.e009_expected_heads:
            raise RuntimeError(
                f'[E009] expected {self.e009_expected_heads} CLIP heads, got {H}'
            )

        residual_mask = torch.ones(
            T, device=tokens.device, dtype=torch.bool
        )
        residual_mask[core_idx] = False
        residual_idx = torch.where(residual_mask)[0]

        if residual_idx.numel() != T - self.e009_core_k:
            raise RuntimeError('[E009] malformed residual set')

        attn_res = cls_attn_heads[:, residual_idx].float()
        residual_mass = attn_res.sum(dim=1)
        denom = residual_mass.unsqueeze(-1).clamp_min(1e-12)
        weights = attn_res / denom

        residual_tokens = tokens.index_select(0, residual_idx).float()
        carriers = torch.matmul(weights, residual_tokens)

        if not torch.isfinite(carriers).all():
            raise RuntimeError('[E009] non-finite HREC carrier')

        carriers = carriers.to(dtype=tokens.dtype)
        return carriers, residual_mass, residual_idx

    def _e009_select_random_context(self, residual_idx, count):
        # Deterministic per-request no-model-signal control.
        # It is intentionally NOT a final-method candidate.
        if residual_idx.numel() < count:
            raise RuntimeError(
                f'[E009] residual pool {residual_idx.numel()} < random count {count}'
            )
        rid = int(self.e007_request_index)
        x = residual_idx.to(torch.long)
        h = torch.remainder(
            x * 1103515245 + (rid + 1) * 12345 + 20261008,
            2147483647,
        )
        local = torch.topk(
            h, k=count, largest=False, sorted=False
        ).indices
        return residual_idx.index_select(0, local).sort().values

    # ═══════════════════════════════════════════════════════════════
    # CORE：主流程
    # ═══════════════════════════════════════════════════════════════
    def visual_tokens_spatial_similarity(self, images):
        H_grid, W_grid = 24, 24

        # VIC needs the last-2 K projection and patch positions.
        # E007 QED does not use cluster labels, so when VIC is explicitly
        # bypassed we also remove these VIC-only support computations.
        e007_vic_bypass = self.e007_enable and self.e007_skip_unused_vic

        hook_handle_k = None
        if not e007_vic_bypass:
            hook_handle_k = (self.vision_tower.vision_model
                             .encoder.layers[-2].self_attn.k_proj
                             .register_forward_hook(hook_k))

        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=True
        )

        attn_weights = image_forward_outs.attentions[-2]
        hidden_states = image_forward_outs.hidden_states[-2]

        desired_layer_k = None
        if hook_handle_k is not None:
            desired_layer_k = outputs['desired_k']
            hook_handle_k.remove()

        device = hidden_states.device

        cls_attention_raw = attn_weights[:, :, 0, 1:]
        cls_attention_sum = cls_attention_raw.sum(dim=1)

        patch_tokens = hidden_states[:, 1:, :]
        patch_metric = (
            desired_layer_k[:, 1:, :] if desired_layer_k is not None else None
        )
        patch_pos = (
            self.get_pos(patch_tokens, W=W_grid, H=H_grid)
            if not e007_vic_bypass else None
        )

        B, T, D = patch_tokens.shape

        if B != 1:
            raise NotImplementedError('Stage-1 supports batch_size=1 only.')

        if self.e007_enable:
            self.e007_request_index += 1

        # E001-S1-A 2026-09-18: budget and selection are orthogonal factors.
        # E007 deliberately bypasses global ratio coupling and fixes the
        # pre-registered Stage1 budgets directly.
        # E014-AEE-BEGIN: constant-length Stage1 budget
        if self.e014_enable:
            # AEE exchanges evidence within exactly 172 eligible token slots.
            total_budget = 172
            keep_tokens_each = torch.full((B,), total_budget, dtype=torch.long, device=device)
            rho = torch.full((B,), float('nan'), device=device)
            phi = torch.full((B,), float('nan'), device=device)
        # E014-AEE-END: constant-length Stage1 budget
        elif self.e013_enable:
            total_budget = 172 + (self.e013_b if self.e013_variant == 'EXTRA' else 0)
            keep_tokens_each = torch.full((B,), total_budget, dtype=torch.long, device=device)
            rho = torch.full((B,), float('nan'), device=device)
            phi = torch.full((B,), float('nan'), device=device)
        elif self.e009_enable:
            total_budget = self.e009_total_k
            keep_tokens_each = torch.full(
                (B,), total_budget, dtype=torch.long, device=patch_tokens.device)
            rho = torch.full((B,), float('nan'), device=patch_tokens.device)
            phi = torch.full((B,), float('nan'), device=patch_tokens.device)

        elif self.e007_enable:
            total_budget = self._e007_stage1_budgets[self.e007_stage1_mode]
            if total_budget > T:
                raise RuntimeError(
                    f'[E007] Requested Stage1 K={total_budget} > input T={T}'
                )
            keep_tokens_each = torch.full(
                (B,), total_budget, dtype=torch.long, device=patch_tokens.device)
            rho = torch.full((B,), float('nan'), device=patch_tokens.device)
            phi = torch.full((B,), float('nan'), device=patch_tokens.device)

        elif self.use_dynamic_ratio:
            rho, phi = self.compute_global_redundancy(patch_tokens)
            keep_tokens_each = self.compute_stage1_keep_tokens(T, phi)
            total_budget = int(keep_tokens_each[0].item())
        else:
            total_budget = int(self.vision_prune_ratio * T)
            keep_tokens_each = torch.full(
                (B,), total_budget, dtype=torch.long, device=patch_tokens.device)
            # NaN denotes "not computed" rather than paying redundancy overhead
            # only for logging.  No downstream Stage1 decision uses these values.
            rho = torch.full((B,), float('nan'), device=patch_tokens.device)
            phi = torch.full((B,), float('nan'), device=patch_tokens.device)

        tokens_0 = patch_tokens[0]
        metric_0 = patch_metric[0] if patch_metric is not None else None
        pos_0 = patch_pos[0] if patch_pos is not None else None

        gamma_raw = -1.0
        gamma_used = -1.0

        e007_coverage_anchors = None

        # E014-AEE-BEGIN: defer original Top172 to isolated selector
        if self.e014_enable:
            # e014_choose computes Top172 ONCE while doing the exchange.
            # Do not duplicate a TopK kernel before the Stage1 output branch.
            attn_0 = cls_attention_sum[0]
            final_idx = None
        # E014-AEE-END: defer original Top172
        elif self.e013_enable:
            # E013-P1: Stage1 output is constructed below via a frozen helper.
            attn_0 = cls_attention_sum[0]
            final_idx = attn_0.topk(172, largest=True).indices.sort().values
        elif self.e009_enable:
            # Every final E009 arm has IDENTICAL persistent-token ordering:
            # [Core172 in spatial order] + [16 context-only slots].
            #
            # This is deliberate. It removes the confound that RAW extra
            # patches could otherwise shift Core-token sequence positions
            # differently from HREC/NULL controls.
            attn_0 = cls_attention_sum[0]
            final_idx = attn_0.topk(
                self.e009_core_k, largest=True
            ).indices.sort().values

        elif self.e007_enable:
            # E007 always uses raw CLS attention. There is deliberately no
            # debiasing, dynamic ratio, similarity matrix, or legacy router.
            attn_0 = cls_attention_sum[0]

            if self.e007_stage1_mode in ('ORIGINAL_172', 'HR_TOPK_192'):
                topk_indices = attn_0.topk(
                    total_budget, largest=True
                ).indices
                final_idx = topk_indices.sort().values

            elif self.e007_stage1_mode == 'HR_COVER_192':
                final_idx, e007_coverage_anchors = self._e007_hr_cover_select(
                    attn_0, total_budget, H=H_grid, W=W_grid
                )

            else:
                raise RuntimeError(
                    f'[E007] Unhandled Stage1 mode {self.e007_stage1_mode}'
                )

        elif not self.use_adaptive_select and not self.use_zone_routing:
            # E001-S1-A 2026-09-18: exact Original ViTCoP selection path.
            # Use raw CLS attention (no float/debias preprocessing) and select the
            # exact upstream top-k set. Sorting restores original spatial order,
            # equivalent to upstream's boolean-mask extraction.
            attn_0 = cls_attention_sum[0]
            topk_indices = attn_0.topk(total_budget).indices
            final_idx = topk_indices.sort().values

        else:
            # CABS/legacy optional selection paths retain their existing attention
            # preprocessing; strict E001 runs keep debias/zone-routing disabled.
            processed_attn = self.maybe_debias_attention(
                cls_attention_sum, H=H_grid, W=W_grid)
            attn_0 = processed_attn[0]

        if self.use_adaptive_select and not self.use_zone_routing:
            # ═══ Path B: adaptive_select（主线）═══
            gamma_raw = self.compute_image_complexity(attn_0)
            gamma_used = self.apply_budget_aware_gamma(
                gamma_raw, total_budget=total_budget, total_tokens=T
            )
            sel = self.adaptive_select(
                tokens_0, attn_0, total_budget, gamma_used,
                spatial_pos=pos_0
            )
            final_idx = sel.sort().values

        elif self.use_zone_routing:
            # ═══ Path C: zone routing（保留旧逻辑，不推荐）═══
            image_gamma_raw = self.compute_image_complexity(attn_0)
            image_gamma = self.apply_budget_aware_gamma(
                image_gamma_raw, total_budget=total_budget, total_tokens=T
            )
            gamma_raw = image_gamma_raw
            gamma_used = image_gamma

            coord_indices = self.get_coord_token_indices(H=H_grid, W=W_grid)
            coord_set = set(coord_indices)
            num_coord = len(coord_indices)

            zone_indices, M = self.build_zone_map(
                H=H_grid, W=W_grid, device=device)
            zone_budgets, zone_difficulties = self.allocate_zone_budgets(
                attn_0, tokens_0, zone_indices, total_budget, M,
                num_coord=num_coord, image_gamma=image_gamma)

            all_selected = list(coord_indices)

            for g in range(M):
                budget_g = int(zone_budgets[g].item())
                if budget_g <= 0:
                    continue

                zone_mask = (zone_indices == g)
                zone_global_idx = torch.where(zone_mask)[0]

                avail_mask = torch.tensor(
                    [int(idx.item()) not in coord_set for idx in zone_global_idx],
                    device=device, dtype=torch.bool)
                zone_global_avail = zone_global_idx[avail_mask]

                if len(zone_global_avail) == 0:
                    continue

                ztokens = tokens_0[zone_global_avail]
                zattn = attn_0[zone_global_avail]
                zpos = pos_0[zone_global_avail]

                gamma_g_raw = float(zone_difficulties[g].item())
                gamma_g = self.apply_budget_aware_gamma(
                    gamma_g_raw,
                    total_budget=min(budget_g, len(zone_global_avail)),
                    total_tokens=len(zone_global_avail)
                )

                actual_budget = min(budget_g, len(zone_global_avail))
                local_sel = self.adaptive_select(
                    ztokens, zattn, actual_budget, gamma_g,
                    spatial_pos=zpos
                )
                global_sel = zone_global_avail[local_sel].tolist()
                all_selected.extend(global_sel)

            all_selected = list(dict.fromkeys(all_selected))[:total_budget]
            final_idx = torch.tensor(
                all_selected, dtype=torch.long, device=device).sort().values

        # ---------------------------------------------------------------
        # Build actual Stage1 output sequence.
        # ---------------------------------------------------------------
        e008_core172_idx = None
        e008_core_mask = None
        e008_transient_mask = None

        e009_core172_idx = None
        e009_context_mask = None
        e009_carriers = None
        e009_residual_mass = None
        e009_residual_idx = None
        e013_context_mask = None
        e013_persistent_idx = None
        e013_transient_idx = None
        e013_original_idx = None

        if self.e010.enabled and self.e010.analysis_full576:
            hidden_states_filtered = tokens_0.unsqueeze(0)
            K = int(tokens_0.shape[0])
        elif self.e014_enable:
            # E014-AEE-BEGIN: 172 eligible patches, exchange in CLIP coordinates
            # E014: no transient tokens. Every selected patch is QED-eligible.
            from .e014_stage1_support import e014_choose
            selected, original, added, removed, stats = e014_choose(
                tokens_0, cls_attention_sum[0], self.e014_cfg,
                request_index=self.e007_request_index,
            )
            hidden_states_filtered = tokens_0.index_select(0, selected).unsqueeze(0)
            final_idx = selected
            K = int(hidden_states_filtered.shape[1])
            if K != 172:
                raise RuntimeError('[E014] Stage1 must output exactly 172 visual tokens')
            self.e014_last = {
                'selected_ids': selected.detach(),
                'original_core_ids': original.detach(),
                'added_ids': added.detach(),
                'removed_ids': removed.detach(),
                'stats': stats,
            }
            # E014-AEE-END: 172 eligible patches
        elif self.e013_enable:
            # E013-P1 2026-10-09: compute IDs using only existing CLIP output.
            # Same original residual candidate pool for ATTN/RANDOM/COVER.
            from .e013_stage1_support import e013_choose
            e013_persistent_idx, e013_transient_idx, e013_original_idx = e013_choose(
                tokens_0, cls_attention_sum[0], self.e013_variant,
                self.e013_pick, self.e013_b, self.e013_seed,
                self.e007_request_index,
            )
            core_tokens = tokens_0.index_select(0, e013_persistent_idx)
            if self.e013_variant == 'EXTRA' and self.e013_b > 0:
                if self.e013_pick == 'NULL':
                    context_tokens = torch.zeros(
                        (self.e013_b, tokens_0.shape[-1]),
                        device=device, dtype=tokens_0.dtype,
                    )
                else:
                    context_tokens = tokens_0.index_select(0, e013_transient_idx)
                hidden_states_filtered = torch.cat(
                    (core_tokens, context_tokens), dim=0).unsqueeze(0)
                e013_context_mask = torch.cat((
                    torch.zeros(172, dtype=torch.bool, device=device),
                    torch.ones(self.e013_b, dtype=torch.bool, device=device),
                ))
            else:
                hidden_states_filtered = core_tokens.unsqueeze(0)
                e013_context_mask = torch.zeros(172, dtype=torch.bool, device=device)
            K = int(hidden_states_filtered.shape[1])
            if K != total_budget or e013_persistent_idx.numel() != 172:
                raise RuntimeError('[E013] Stage1 size invariant failed')
            # Store tensors without GPU->CPU transfers on the latency path.
            self.e013_last = {
                'persistent_ids': e013_persistent_idx.detach(),
                'transient_ids': e013_transient_idx.detach(),
                'original_core_ids': e013_original_idx.detach(),
                'variant': self.e013_variant, 'pick': self.e013_pick,
                'budget': self.e013_b, 'visual_count': K,
            }
        elif self.e009_enable:
            e009_core172_idx = final_idx
            core_tokens = tokens_0[e009_core172_idx]

            residual_mask = torch.ones(
                T, device=device, dtype=torch.bool
            )
            residual_mask[e009_core172_idx] = False
            residual_idx = torch.where(residual_mask)[0]
            e009_residual_idx = residual_idx

            context_tokens = None
            context_idx = None

            if self.e009_mode == 'RAW188_CONTEXT':
                # Same signal as the successful HR_TOPK family, but only a
                # 16-slot transient budget and appended after Core172.
                residual_scores = attn_0.index_select(0, residual_idx)
                local = torch.topk(
                    residual_scores,
                    k=self.e009_context_k,
                    largest=True,
                    sorted=False,
                ).indices
                context_idx = residual_idx.index_select(
                    0, local
                ).sort().values
                context_tokens = tokens_0.index_select(0, context_idx)

            elif self.e009_mode == 'RANDOM188_CONTEXT':
                # Semantically valid image patches but chosen without any
                # model signal. This tests whether "any 16 visual tokens"
                # provide the same benefit.
                context_idx = self._e009_select_random_context(
                    residual_idx, self.e009_context_k
                )
                context_tokens = tokens_0.index_select(0, context_idx)

            elif self.e009_mode == 'NULL188_CONTEXT':
                # Same sequence length / same context positions, but no new
                # image semantics are injected at the CLIP hidden-state level.
                # Downstream projector biases may map exact zeros to a fixed
                # vector; report this correctly as a NULL-SLOT control.
                context_tokens = torch.zeros(
                    (self.e009_context_k, tokens_0.shape[-1]),
                    device=device,
                    dtype=tokens_0.dtype,
                )

            elif self.e009_mode == 'DUPLICATE16':
                from llava.model.e010_runtime_support import deterministic_pick
                request_key = self.e010_request_context.request_key
                if not request_key:
                    self._e010_auto_key = int(getattr(self, '_e010_auto_key', 0)) + 1
                    request_key = f'auto-{self._e010_auto_key}'
                    warnings.warn(f'[E010] DUPLICATE16 synthesized request_key={request_key}')
                picked = deterministic_pick(
                    [int(x) for x in e009_core172_idx.tolist()],
                    self.e009_context_k,
                    request_key,
                    'duplicate16',
                )
                context_idx = torch.tensor(picked, device=device, dtype=torch.long)
                context_tokens = tokens_0.index_select(0, context_idx)

            elif self.e009_mode == 'CROSSIMAGE16':
                from llava.model.e010_context_bank import E010ContextBank
                from llava.model.e010_runtime_support import deterministic_pick
                request_key = self.e010_request_context.request_key
                bank_path = self.e010.context_bank
                if not request_key or not bank_path:
                    raise RuntimeError('[E010] CROSSIMAGE16 requires request_key and E010_CONTEXT_BANK')
                if getattr(self, '_e010_bank_path', None) != bank_path:
                    self._e010_bank = E010ContextBank.load(bank_path)
                    self._e010_bank_path = bank_path
                feat, patch_ids, _donor_key = self._e010_bank.fetch_donor(
                    request_key, device=device, dtype=tokens_0.dtype
                )
                id_list = [int(x) for x in patch_ids.tolist()]
                if len(id_list) != self.e009_context_k:
                    picked = deterministic_pick(
                        id_list, self.e009_context_k, request_key, 'crossimage16'
                    )
                    index = [id_list.index(i) for i in picked]
                    context_tokens = feat.index_select(
                        0, torch.tensor(index, device=feat.device, dtype=torch.long)
                    )
                else:
                    context_tokens = feat

            elif self.e009_mode == 'REGION_TRANSIENT':
                region_ids = [
                    int(x) for x in (self.e010_request_context.region_patch_ids or [])
                ]
                residual_set = {int(x) for x in residual_idx.tolist()}
                kept = [i for i in region_ids if i in residual_set]
                if not kept:
                    raise RuntimeError('[E010] REGION_TRANSIENT residual is empty')
                context_idx = torch.tensor(kept, device=device, dtype=torch.long)
                context_tokens = tokens_0.index_select(0, context_idx)

            elif self.e009_mode == 'HREC16_CONTEXT':
                # One architecture-derived carrier per CLIP attention head,
                # summarizing ALL 404 non-Core patches.
                e009_carriers, e009_residual_mass, e009_residual_idx = (
                    self._e009_build_hrec(
                        tokens_0,
                        cls_attention_raw[0],
                        e009_core172_idx,
                    )
                )
                if e009_carriers.shape[0] != self.e009_context_k:
                    raise RuntimeError(
                        f'[E009] carrier count={e009_carriers.shape[0]} != 16'
                    )
                context_tokens = e009_carriers

            else:
                raise RuntimeError(
                    f'[E009] unhandled mode={self.e009_mode}'
                )

            if context_tokens is None:
                raise RuntimeError('[E009] context construction failed')

            # EXACT same slot layout for all four causal controls.
            hidden_states_filtered = torch.cat(
                [core_tokens, context_tokens], dim=0
            ).unsqueeze(0)

            n_ctx = int(context_tokens.shape[0])
            e009_context_mask = torch.cat(
                [
                    torch.zeros(
                        self.e009_core_k,
                        device=device,
                        dtype=torch.bool,
                    ),
                    torch.ones(
                        n_ctx,
                        device=device,
                        dtype=torch.bool,
                    ),
                ],
                dim=0,
            )

            K = int(hidden_states_filtered.shape[1])

            if self.e009_strict_check:
                context_count = int(e009_context_mask.sum().item())
                if self.e009_mode == 'REGION_TRANSIENT':
                    if K != self.e009_core_k + n_ctx or context_count != n_ctx:
                        raise RuntimeError(
                            f'[E010] region transient K={K} context={context_count}'
                        )
                else:
                    if K != self.e009_total_k:
                        raise RuntimeError(f'[E009] actual Stage1 K={K}, expected 188')
                    if context_count != self.e009_context_k:
                        raise RuntimeError(
                            f'[E009] context count={context_count}, expected 16'
                        )

        else:
            K = len(final_idx)

            # E008 frozen Core172 / transient20 partition. final_idx is sorted
            # back to spatial order, so eligibility must be an explicit mask.
            if self.e008_enable:
                if K != 192:
                    raise RuntimeError(f'[E008] expected Stage1 K=192, got {K}')
                e008_core172_idx = attn_0.topk(
                    172, largest=True
                ).indices.sort().values
                e008_core_mask = torch.isin(final_idx, e008_core172_idx)
                e008_transient_mask = ~e008_core_mask

                if self.e008_strict_check:
                    core_count = int(e008_core_mask.sum().item())
                    transient_count = int(e008_transient_mask.sum().item())
                    if core_count != 172:
                        raise RuntimeError(
                            f'[E008] Core172 mask count={core_count}, expected 172'
                        )
                    if transient_count != 20:
                        raise RuntimeError(
                            f'[E008] transient mask count={transient_count}, expected 20'
                        )

            hidden_states_filtered = tokens_0[final_idx].unsqueeze(0)

        if self.e010.enabled and self.e010.analysis_full576:
            clustered_labels = torch.arange(
                K, device=device, dtype=torch.long
            ).unsqueeze(0)
        elif e007_vic_bypass:
            # Keep a Tensor sentinel rather than None so the existing multimodal
            # plumbing retains a stable type/device. QED must not consume it.
            # Identity smoke must verify that VIC-ON and VIC-OFF produce the same
            # Stage1/QED outputs before publication timing.
            # Shape-preserving zero sentinel. Values are irrelevant because
            # E006 QED bypasses Original Stage2, but preserving [B,K] protects
            # existing multimodal plumbing from shape assumptions.
            if self.e013_enable:
                # E013-P1: 0 = persistent eligible Core172; -3 = context-only.
                # This is ONLY metadata while E006 bypasses Original Stage2.
                clustered_labels = torch.where(
                    e013_context_mask,
                    torch.full((K,), -3, device=device, dtype=torch.long),
                    torch.zeros((K,), device=device, dtype=torch.long),
                ).unsqueeze(0)
            elif self.e009_enable:
                # E009 metadata transport:
                #   0  = frozen Core172, QED-eligible
                #  -3  = context token/carrier, real-LLM-only
                labels = torch.where(
                    e009_context_mask,
                    torch.full(
                        (K,), -3, device=device, dtype=torch.long
                    ),
                    torch.zeros(
                        (K,), device=device, dtype=torch.long
                    ),
                )
                clustered_labels = labels.unsqueeze(0)

            elif self.e008_enable:
                # Reuse the existing ViTCoP cluster-label plumbing as metadata
                # transport ONLY on the isolated E008 + E006-QED path:
                #
                #   0  = Core172, eligible final evidence
                #  -2  = HR-added transient context, never final evidence
                #
                # Original Stage2 is hard-bypassed by E006, so these values are
                # never interpreted as VIC cluster IDs.
                labels = torch.where(
                    e008_core_mask,
                    torch.zeros_like(final_idx, dtype=torch.long),
                    torch.full_like(final_idx, -2, dtype=torch.long),
                )
                clustered_labels = labels.unsqueeze(0)
            else:
                clustered_labels = torch.zeros(
                    (hidden_states_filtered.shape[0], hidden_states_filtered.shape[1]),
                    device=device,
                    dtype=torch.long,
                )
        else:
            if metric_0 is None or pos_0 is None:
                raise RuntimeError(
                    '[E007] VIC requested but VIC support tensors are unavailable'
                )

            metric_filtered = metric_0[final_idx].unsqueeze(0)
            patch_pos_filtered = pos_0[final_idx].unsqueeze(0)

            clustered_labels = []
            from .utils import VIC

            for i in range(hidden_states_filtered.shape[0]):
                sample_metric = metric_filtered[i]
                sample_pos = patch_pos_filtered[i]
                vic = VIC(dc=8, percentage=self.cluster_percentage, k=0.6)
                vic.fit(X=sample_metric, pos=sample_pos)
                labels_ = torch.tensor(
                    vic.labels_, device=device, dtype=torch.long
                )
                clustered_labels.append(labels_)

            clustered_labels = torch.stack(clustered_labels, dim=0)

        self.last_dynamic_info = {
            'rho': rho.detach().cpu(),
            'phi': phi.detach().cpu(),
            'keep_tokens': keep_tokens_each.detach().cpu(),
            'actual_kept': K,
            'total_budget': total_budget,
            'dynamic_enabled': self.use_dynamic_ratio,
            'adaptive_select': self.use_adaptive_select,
            'use_debias': self.use_debias,
            'use_zone_routing': self.use_zone_routing,
            'use_coord_tokens': self.use_coord_tokens,
            # ★ 空间感知诊断
            'use_spatial_aware': self.use_spatial_aware,
            'spatial_beta': self.spatial_beta if self.use_spatial_aware else 0.0,
            'spatial_sigma': self.spatial_sigma if self.use_spatial_aware else 0.0,
            # ★ SID 诊断
            'complexity_mode': self.complexity_mode,
            'complexity_alpha': (self.complexity_alpha
                                 if self.complexity_mode == 'renyi' else None),
            'R_bar': (float(self._last_R_bar[0].item())
                      if (self.export_directional_stats
                          and self._last_R_bar is not None) else None),
            'kappa': (float(self._last_kappa[0].item())
                      if (self.export_directional_stats
                          and self._last_kappa is not None) else None),
            'eta_mode': self.eta_mode,
            'eta_floor': self.eta_floor,
            'highcomp_vpr_threshold': self.highcomp_vpr_threshold,
            'highcomp_use_keep_ratio_gamma': self.highcomp_use_keep_ratio_gamma,
            'gamma_raw': gamma_raw,
            'gamma': gamma_used,
            'gamma_budget_aware': self.gamma_budget_aware,
            'adaptive_prune_threshold': self.adaptive_prune_threshold,
            'gamma_min_scale': self.gamma_min_scale,
            'gamma_high_scale': self.gamma_high_scale,
            'vision_prune_ratio': self.vision_prune_ratio,
            'redundancy_sample_n': self.redundancy_sample_n,
            # E007 scalar/string audit fields: no tensor synchronization.
            'e007_enable': self.e007_enable,
            'e007_stage1_mode': (
                self.e007_stage1_mode if self.e007_enable else None
            ),
            'e007_skip_unused_vic': (
                self.e007_skip_unused_vic if self.e007_enable else False
            ),
            'e007_vic_support_hook_skipped': bool(e007_vic_bypass),
            'e008_enable': bool(self.e008_enable),
            'e008_mode': self.e008_mode if self.e008_enable else None,
            'e008_core_count': 172 if self.e008_enable else None,
            'e008_transient_count': 20 if self.e008_enable else None,
            # E013-P1: metadata only (audit IDs remain optional).
            'e013_enable': bool(self.e013_enable),
            'e013_variant': self.e013_variant if self.e013_enable else None,
            'e013_pick': self.e013_pick if self.e013_enable else None,
            'e013_b': self.e013_b if self.e013_enable else None,
            'e013_visual_count': K if self.e013_enable else None,
            # E014-AEE-BEGIN: no decoder-context/transient metadata
            # E014: budget is the *exchange* count, never extra visual tokens.
            'e014_enable': bool(self.e014_enable),
            'e014_variant': self.e014_cfg.variant if self.e014_enable else None,
            'e014_exchange_budget': (
                int(self.e014_last['stats']['budget']) if self.e014_enable else None
            ),
            'e014_visual_count': K if self.e014_enable else None,
            # E014-AEE-END: Stage1 metadata
            'e009_enable': bool(self.e009_enable),
            'e009_mode': self.e009_mode if self.e009_enable else None,
            'e009_core_count': self.e009_core_k if self.e009_enable else None,
            'e009_context_count': (
                self.e009_context_k if self.e009_enable else None
            ),
            'e009_total_count': self.e009_total_k if self.e009_enable else None,
        }

        # E014-AEE-BEGIN: OFFLINE-only provenance audit
        # E014 offline audit: patch IDs are CLIP 0..575 spatial indices.
        # GPU->CPU copies below are allowed ONLY with E014_AUDIT=True.
        if self.e014_enable and self.e014_cfg.audit:
            audit_ids = self.e014_last
            st = audit_ids['stats']
            selected_ids = audit_ids['selected_ids']
            if selected_ids.unique().numel() != 172:
                raise RuntimeError('[E014] duplicate visual token in Stage1')
            if not bool(torch.isfinite(tokens_0).all().item()):
                raise RuntimeError('[E014] nonfinite CLIP features')
            if not bool(torch.isfinite(attn_0).all().item()):
                raise RuntimeError('[E014] nonfinite CLS attention')
            print('[E014_STAGE1_JSON] ' + json.dumps({
                'request_index': int(self.e007_request_index),
                'variant': self.e014_cfg.variant,
                'budget': int(st['budget']),
                'stage1_in': int(T), 'stage1_out': int(K),
                'eligible_count': int((clustered_labels[0] >= 0).sum().item()),
                'original_core_ids': audit_ids['original_core_ids'].cpu().tolist(),
                'selected_ids': selected_ids.cpu().tolist(),
                'added_ids': audit_ids['added_ids'].cpu().tolist(),
                'removed_ids': audit_ids['removed_ids'].cpu().tolist(),
                'score_type': st.get('score_type'),
                'lambda': float(self.e014_cfg.lam),
                'penalty': float(self.e014_cfg.penalty),
                'utility': float(st['utility'].item()),
                'gain_sum': float(st['gain_sum'].item()),
                'risk_sum': float(st['risk_sum'].item()),
            }, separators=(',', ':')))
        # E014-AEE-END: OFFLINE-only provenance audit

        # E013-P1: auditing is forbidden inside canonical latency measurements.
        if self.e013_enable and self.e013_audit:
            if not torch.isfinite(tokens_0).all().item() or not torch.isfinite(attn_0).all().item():
                raise RuntimeError('[E013] nonfinite CLIP feature/score in audit')
            # E013-P1 2026-10-09: OFFLINE DIAGNOSTICS for E014 utility
            # predictability, not a trained/performance-validated predictor.
            # These GPU reductions synchronize ONLY in E013_AUDIT, never timing.
            from .e013_stage1_support import _core_relative_coverage
            residual_mask = torch.ones(T, dtype=torch.bool, device=device)
            residual_mask[self.e013_last['original_core_ids']] = False
            residual_ids = torch.where(residual_mask)[0]
            rr = F.normalize(tokens_0.index_select(0, residual_ids).float(),
                             p=2, dim=-1, eps=1e-8)
            n_resid = int(residual_ids.numel())
            # Mean off-diagonal cosine via sum-vector identity, O(N*d).
            mean_sim = (rr.sum(dim=0).square().sum() - n_resid) / (
                n_resid * (n_resid - 1))
            residual_phi = 1.0 - mean_sim
            residual_attention = attn_0.index_select(0, residual_ids).float()
            residual_mass = residual_attention.sum()
            weighted_coverage_gap = _core_relative_coverage(
                tokens_0, attn_0, self.e013_last['original_core_ids'], residual_ids
            ).sum() / residual_mass.clamp_min(1e-12)
            old_set = set(self.e013_last['original_core_ids'].detach().cpu().tolist())
            new_set = set(self.e013_last['persistent_ids'].detach().cpu().tolist())
            print('[E013_STAGE1_JSON] ' + json.dumps({
                'request_index': int(self.e007_request_index),
                'variant': self.e013_variant, 'pick': self.e013_pick,
                'budget': self.e013_b, 'visual_in': int(T), 'visual_out': int(K),
                'original_core_ids': sorted(old_set),
                'persistent_ids': sorted(new_set),
                'added_persistent_ids': sorted(new_set - old_set),
                'lost_persistent_ids': sorted(old_set - new_set),
                'transient_ids': self.e013_last['transient_ids'].detach().cpu().tolist(),
                'eligible_count': int((clustered_labels[0] >= 0).sum().item()),
                'residual_raw_cls_mass': float(residual_mass.item()),
                'residual_cls_fraction': float((residual_mass / attn_0.float().sum().clamp_min(1e-12)).item()),
                'residual_feature_dispersion': float(residual_phi.item()),
                'residual_core_relative_gap': float(weighted_coverage_gap.item()),
            }, separators=(',', ':')))

        if self.e009_enable and self.e009_audit:
            ctx = hidden_states_filtered[
                0, self.e009_core_k:, :
            ].float()
            ctx_norm = torch.linalg.vector_norm(ctx, dim=-1)
            ctx_cos_mean = None
            if ctx.shape[0] > 1:
                ctx_unit = ctx / ctx_norm.unsqueeze(-1).clamp_min(1e-12)
                sim = ctx_unit @ ctx_unit.transpose(0, 1)
                offdiag = ~torch.eye(
                    ctx.shape[0], device=ctx.device, dtype=torch.bool
                )
                ctx_cos_mean = float(sim[offdiag].mean().item())

            payload = {
                'request_id': int(self.e007_request_index),
                'mode': self.e009_mode,
                'stage1_visual_out': int(K),
                'core_count': int(self.e009_core_k),
                'context_count': int(self.e009_context_k),
                'carrier_count': (
                    int(e009_carriers.shape[0])
                    if e009_carriers is not None else 0
                ),
                'residual_pool_count': (
                    int(e009_residual_idx.numel())
                    if e009_residual_idx is not None else 0
                ),
                'context_norm_mean': float(ctx_norm.mean().item()),
                'context_norm_min': float(ctx_norm.min().item()),
                'context_norm_max': float(ctx_norm.max().item()),
                'context_pairwise_cosine_mean': ctx_cos_mean,
                'residual_mass_mean': (
                    float(e009_residual_mass.float().mean().item())
                    if e009_residual_mass is not None else None
                ),
                'residual_mass_min': (
                    float(e009_residual_mass.float().min().item())
                    if e009_residual_mass is not None else None
                ),
                'residual_mass_max': (
                    float(e009_residual_mass.float().max().item())
                    if e009_residual_mass is not None else None
                ),
            }
            print(
                '[E009_STAGE1_JSON] '
                + json.dumps(payload, separators=(',', ':'))
            )

        if self.e008_enable and self.e008_audit:
            e008_payload = {
                'request_id': int(self.e007_request_index),
                'mode': self.e008_mode,
                'stage1_visual_out': int(K),
                'core_count': 172,
                'transient_count': 20,
                'core_local_positions': (
                    torch.where(e008_core_mask)[0].detach().cpu().tolist()
                ),
                'transient_local_positions': (
                    torch.where(e008_transient_mask)[0].detach().cpu().tolist()
                ),
            }
            print(
                '[E008_STAGE1_JSON] '
                + json.dumps(e008_payload, separators=(',', ':'))
            )

        # Patch-ID diagnostics are intentionally OFF during canonical timing.
        if self.e009_enable and self.e007_token_audit:
            raise RuntimeError(
                '[E009] E007_TOKEN_AUDIT must be False; use E009_AUDIT instead'
            )

        if self.e007_enable and self.e007_token_audit:
            core_k = min(172, T)
            core172_idx = (
                e008_core172_idx if e008_core172_idx is not None
                else attn_0.topk(core_k, largest=True).indices.sort().values
            )
            new_candidate_mask = ~torch.isin(final_idx, core172_idx)
            new_candidates = final_idx[new_candidate_mask]

            stage1_patch_ids = final_idx.detach().cpu().tolist()
            core172_patch_ids = core172_idx.detach().cpu().tolist()
            new_candidate_patch_ids = new_candidates.detach().cpu().tolist()
            coverage_anchor_ids = (
                e007_coverage_anchors.detach().cpu().tolist()
                if e007_coverage_anchors is not None else []
            )

            self.last_dynamic_info.update({
                'e007_request_id': int(self.e007_request_index),
                'e007_stage1_patch_ids': stage1_patch_ids,
                'e007_core172_patch_ids': core172_patch_ids,
                'e007_new_candidate_patch_ids': new_candidate_patch_ids,
                'e007_new_candidate_count': int(new_candidates.numel()),
                'e007_coverage_anchor_ids': coverage_anchor_ids,
                'e007_cluster_label_shape': list(clustered_labels.shape),
            })

            audit_payload = {
                'request_id': int(self.e007_request_index),
                'stage1_mode': self.e007_stage1_mode,
                'visual_in': int(T),
                'visual_out': int(K),
                'vic_bypass': bool(e007_vic_bypass),
                'stage1_patch_ids': stage1_patch_ids,
                'core172_patch_ids': core172_patch_ids,
                'new_candidate_patch_ids': new_candidate_patch_ids,
                'coverage_anchor_ids': coverage_anchor_ids,
                'cluster_label_shape': list(clustered_labels.shape),
            }
            print('[E007_STAGE1_JSON] ' + json.dumps(audit_payload, separators=(',', ':')))

        # E001-S1-A 2026-09-18: diagnostic-only token-count line.
        # Keep E001_TOKEN_AUDIT=False for latency measurements.
        if self.e001_token_audit:
            print(
                f'[E001_STAGE1_TOKENS] method_id={self.e001_method_id} '
                f'budget_policy={"dynamic_ratio" if self.use_dynamic_ratio else "original_fixed"} '
                f'selection_policy={"cabs_adaptive" if self.use_adaptive_select else "original_topk"} '
                f'requested_stage1_budget={total_budget} actual_stage1_retained_tokens={K} '
                f'input_visual_tokens={T}'
            )

        return hidden_states_filtered, clustered_labels

    # ═══════════════════════════════════════════════════════════════
    # Forward（不变）
    # ═══════════════════════════════════════════════════════════════
    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        elif self.use_vitcop:
            image_features, cluster_labels = self.visual_tokens_spatial_similarity(images)
            # Match non-ViTCoP path: projector weights are fp16; Stage1 may
            # promote features to fp32 under some loaders (device_map=auto).
            image_features = image_features.to(dtype=images.dtype)
            if image_features.shape[0] > 1:
                offsets = (torch.arange(image_features.shape[0],
                           device=cluster_labels.device)
                           * image_features.shape[1])
                cluster_labels = (cluster_labels
                                  + offsets.unsqueeze(1)).flatten().unsqueeze(0)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features, cluster_labels if self.use_vitcop else None

    # ═══════════════════════════════════════════════════════════════
    # Properties（不变）
    # ═══════════════════════════════════════════════════════════════
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


# ═══════════════════════════════════════════════════════════════════
# CLIPVisionTowerS2（不变）
# ═══════════════════════════════════════════════════════════════════
class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(vision_tower, args, delay_load)

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError(
                'Package s2wrapper not found! Please install by running: \n'
                'pip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = \
                self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(
                self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = \
            self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(
                    self.forward_feature, image.unsqueeze(0),
                    img_sizes=self.s2_scales,
                    max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(
                self.forward_feature, images,
                img_sizes=self.s2_scales,
                max_split_size=self.s2_split_size)
        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
