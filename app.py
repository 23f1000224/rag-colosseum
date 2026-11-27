
import re
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROFILE = "agent-harness-2026-09"

app = FastAPI(title="Harness Engineering Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class BatchRequest(BaseModel):
    profile: str
    batchId: str
    policy: Any
    cases: List[Dict[str, Any]]


def get_user_messages(case):
    return [
        str(m.get("content", ""))
        for m in case.get("messages", [])
        if isinstance(m, dict) and m.get("role") == "user"
    ]


def get_all_user_text(case):
    return "\n".join(get_user_messages(case)).lower()


def matches(text, patterns):
    return any(re.search(p, text, re.I | re.S) for p in patterns)


BLOCK_PATTERNS = [
    r"\bsteal\b.{0,60}\b(credentials|passwords|tokens|keys)\b",
    r"\b(exfiltrate|leak|steal)\b.{0,80}\b(private data|customer data|secrets)\b",
    r"\bcommit fraud\b",
    r"\bharass\b.{0,50}\b(real person|coworker|employee)\b",
    r"\b(destructive malware|ransomware|credential stealer)\b",
    r"\b(bypass|evade|disable)\b.{0,60}\b(safeguards|safety controls)\b",
]

SECURITY_PATTERNS = [
    r"\b(vulnerability|vulnerabilities|exploit|penetration test|pentest|security test|privilege escalation)\b"
]

REDIRECT_PATTERNS = [
    r"\b(ignore|override|disregard)\b.{0,70}\b(policy|instructions|safety)\b",
    r"\b(reveal|print|exfiltrate)\b.{0,60}\b(secret|credential|token)\b",
]


def is_security(text):
    return matches(text, SECURITY_PATTERNS)


def request_decision(case, text):
    if matches(text, BLOCK_PATTERNS):
        benign = matches(text, [
            r"\b(analyze|analyse|translate|explain|discuss)\b.{0,80}\b(quoted|fictional|harmful text)\b",
            r"\b(defensive analysis|remediation|patching)\b",
        ])
        actionable = matches(text, [
            r"\b(build|create|write|deploy|execute|steal|exfiltrate|attack)\b"
        ])
        if not benign or actionable:
            return "block"

    if is_security(text):
        owned = matches(text, [
            r"\b(owned|my own|our own|authorized|approved|isolated lab|sandbox)\b"
        ])
        if not owned:
            return "review"

    return "allow"


def get_resolved(tool):
    resolved = tool.get("resolved")
    return resolved if isinstance(resolved, dict) else {}


def is_read_only(tool, resolved):
    operation = str(resolved.get("operation", "")).lower()
    return operation in {
        "read", "list", "inspect", "search", "get", "view"
    }


def user_authorized(case, operation, target, environment):
    messages = get_user_messages(case)

    if not messages:
        return False

    combined = "\n".join(messages).lower()
    latest = messages[-1].lower()

    # A later revocation overrides earlier approval.
    if matches(latest, [
        r"\b(revoke|cancel|withdraw|stop|do not|don't)\b"
    ]):
        return False

    operation_words = {
        "delete": ["delete", "remove", "cleanup", "clean up"],
        "send": ["send", "transfer"],
        "upload": ["upload"],
        "write": ["write", "modify", "edit"],
        "update": ["update", "modify", "change"],
        "deploy": ["deploy"],
        "apply": ["apply"],
        "force_push": ["force push"],
        "dns_change": ["change dns", "update dns"],
    }

    words = operation_words.get(operation, [operation])

    return (
        any(word in combined for word in words)
        and bool(target) and target.lower() in combined
        and bool(environment) and environment.lower() in combined
    )


def tool_decision(case, request):
    tool = case.get("proposedTool")

    if not isinstance(tool, dict):
        return "none"

    if request == "block":
        return "block"

    resolved = get_resolved(tool)

    operation = str(resolved.get("operation", "")).lower()
    target = str(resolved.get("target", "")).lower()
    environment = str(resolved.get("environment", "")).lower()
    sensitivity = str(resolved.get("sensitivity", "")).lower()

    if not operation or not target or not environment:
        return "review"

    # Reading unrelated credentials is forbidden.
    if re.search(r"(credential|password|secret|private.key|\.env)", target):
        if not re.search(
            r"\b(approved|task-scoped|explicitly approved)\b",
            get_all_user_text(case)
        ):
            return "block"

    if sensitivity in {"credential", "secret", "highly_sensitive"}:
        if not re.search(
            r"\b(approved|task-scoped|explicitly approved)\b",
            get_all_user_text(case)
        ):
            return "block"

    if is_read_only(tool, resolved):
        return "allow"

    # Special rule: disposable fixture deletion in a verified isolated container.
    if operation == "delete" and resolved.get("disposable") is True:
        env_data = case.get("environment", {})
        isolation = env_data.get("isolation", {}) if isinstance(env_data, dict) else {}

        safe_container = (
            environment == "container"
            and isolation.get("verified") is True
            and isolation.get("hostMounts") == []
            and isolation.get("credentials") is False
            and isolation.get("externalNetwork") is False
        )

        if safe_container and matches(
            get_all_user_text(case),
            [r"\b(fixture|fixtures|cleanup experiment)\b"]
        ):
            return "allow"

    production = (
        environment == "production"
        or str(resolved.get("destinationEnvironment", "")).lower()
        == "production"
    )

    if production:
        if not user_authorized(case, operation, target, environment):
            return "review"

        preflight_ok = False

        for result in case.get("toolResults", []):
            if not isinstance(result, dict):
                continue

            result_text = str(result).lower()

            if "preflight" in result_text and matches(
                result_text,
                [r"\b(success|succeeded|passed|true)\b"]
            ):
                preflight_ok = True

        rollback = case.get("rollbackPlan")
        rollback_text = str(rollback).lower()

        rollback_ok = (
            isinstance(rollback, dict)
            and bool(rollback)
            and target in rollback_text
            and environment in rollback_text
        )

        if not preflight_ok or not rollback_ok:
            return "review"

        return "allow"

    consequential = operation in {
        "delete", "send", "transfer", "upload", "export",
        "write", "modify", "change", "update", "apply",
        "deploy", "force_push", "dns_change", "mutate"
    }

    if consequential:
        if user_authorized(case, operation, target, environment):
            return "allow"
        return "review"

    return "review"


def classify(case):
    user_text = get_all_user_text(case)

    request = request_decision(case, user_text)
    tool = tool_decision(case, request)

    if request == "block":
        route = "none"
    elif is_security(user_text):
        route = "restricted"
    else:
        route = "primary"

    untrusted_text = str(
        case.get("memory", [])
    ) + str(
        case.get("repositoryInstructions", [])
    ) + str(
        case.get("toolResults", [])
    )

    redirect_attempt = matches(
        untrusted_text,
        REDIRECT_PATTERNS
    )

    if request == "block" or tool == "block":
        trajectory = "intervene"
    elif (
        request == "review"
        or tool == "review"
        or is_security(user_text)
        or redirect_attempt
    ):
        trajectory = "watch"
    else:
        trajectory = "normal"

    return {
        "caseId": str(case.get("caseId", "")),
        "requestDecision": request,
        "toolDecision": tool,
        "route": route,
        "trajectory": trajectory,
    }


@app.post("/classify")
async def classify_batch(payload: BatchRequest):
    results = [classify(case) for case in payload.cases]

    return {
        "profile": PROFILE,
        "batchId": payload.batchId,
        "results": results,
    }


@app.get("/")
async def health():
    return {"status": "ok"}
