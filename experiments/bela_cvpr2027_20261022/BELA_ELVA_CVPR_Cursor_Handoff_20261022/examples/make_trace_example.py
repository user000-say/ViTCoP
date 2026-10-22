"""Generate one synthetic VALID logic audit (not an actual GPU result)."""
from pathlib import Path
import json
from bela_next.budget import PROFILES
s=PROFILES["normal"].schedule()
entry=list(range(s.k0))
row={"sample_id":"SYNTHETIC_LOGIC_TEST_NOT_GPU","schedule":s.as_dict(),
     "entry_patch_ids":entry,
     "events":[
         {"post_block":s.cut1,"selected_patch_ids":entry[:s.kmid]},
         {"post_block":s.cut2,"selected_patch_ids":entry[:s.kfinal]},
     ],
     "observed_visual_counts_per_block":list(s.counts())}
p=Path(__file__).parent/"sample_trace.jsonl"
p.write_text(json.dumps(row)+"\n",encoding="utf-8")
print(p)
