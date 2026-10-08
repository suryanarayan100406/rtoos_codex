# Configuration files

`default.yaml` contains every typed tunable. `profiles/fast.yaml`,
`balanced.yaml`, and `max.yaml` only override those fields and all select the
implemented point-fusion/Poisson/best-view baseline. `models.yaml` is intentionally
empty until a user supplies exact local checkpoint files and measured hashes.

Dataset descriptors belong in `datasets/`; copy `mission.yaml.example`, never put
mission media or checkpoint weights in source control. Descriptor paths resolve
relative to the descriptor file. The typed schema lives in
`src/drishti/config/models.py`; unknown keys fail validation.

Precedence is flags, environment, descriptor, profile, defaults. For environment
variables use `DRISHTI_SECTION__FIELD`, for example:

```bash
DRISHTI_DENSE__VOXEL_M=0.2 drishti run --dataset mission.yaml
```
