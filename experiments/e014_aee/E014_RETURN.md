# E014_RETURN
Generated: 2026-10-10T16:08:56

## Provenance
- git HEAD: `2cb6aa684d033fca6104e97a79cae9b7ec90ad52`
- clip SHA256: `a32c25b07d9e1d4cca2d54c5794585fad9b1e87a03ed95c8d1380b958b37c4af`
- llama SHA256: `d1a41b5a1faa6ed89051684a91641e150d7b5a4e2997fde0e719c41b06763827`
- remote: `fork` → https://github.com/user000-say/ViTCoP
- protocol: LLaVA-1.5-7B FP16 SDPA, Stage1=172, S=D=L8, QED K∈{96,64,32}

## P0
- CPU tests: 12/12 PASS
- compileall: PASS
- E013 base SHA matched before patch; E014 sources installed by byte copy
- GPU smoke arms: 11/11 PASS

- `ADAPTIVE_B_ONLY_K96`: ok=True budgets=[32, 32] errors=[]
- `AEE_K32`: ok=True budgets=[32, 32] errors=[]
- `AEE_K64`: ok=True budgets=[32, 32] errors=[]
- `AEE_K96`: ok=True budgets=[32, 32] errors=[]
- `BASE_K32`: ok=True budgets=[0, 0] errors=[]
- `BASE_K64`: ok=True budgets=[0, 0] errors=[]
- `BASE_K96`: ok=True budgets=[0, 0] errors=[]
- `FIXED_COVER_B16_K32`: ok=True budgets=[16, 16] errors=[]
- `FIXED_COVER_B16_K64`: ok=True budgets=[16, 16] errors=[]
- `FIXED_COVER_B16_K96`: ok=True budgets=[16, 16] errors=[]
- `FIXED_RISK_B16_K96`: ok=True budgets=[16, 16] errors=[]

## Full accuracy

## Judgment (fill after full N)
- Pending full_k96/k64/k32 completion for AGREE/MODIFY/REJECT.
- Note: panel AEE/ADAPTIVE chose b=32 on smoke samples — report histogram on full audit before claiming adaptive success.
