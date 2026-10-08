"""Final report stage; the runner can also call build_report on failure."""

from drishti.report import build_report
from drishti.stages.base import StageResult


class ReportStage:
    name = "report"
    requires = ("s10_export",)
    config_sections = ("report",)

    def run(self, bundle, cfg, runtime):
        outputs = build_report(bundle, cfg)
        report = bundle.read_json(outputs["json"])
        return StageResult(outputs=outputs,
                           metrics={"accuracy_status": report["accuracy"]["status"],
                                    "issues": report["issues"]},
                           confidence=None,
                           degraded_reason="; ".join(report["issues"]) or None)


STAGE = ReportStage()
