#!/usr/bin/env python
"""Build the Pulumi-Python-for-Kubernetes JSONL splits for the small-LLM fine-tune.

Each row pairs a natural-language infrastructure request with a complete Pulumi
``__main__.py`` written in the Hiro-IaC house style (``import pulumi`` +
``import pulumi_kubernetes as k8s``, fully-qualified ``k8s.<group>.<version>.<Type>``
classes, pinned image tags, hardened security context, CPU/memory limits and
requests on every container, one ``pulumi.export`` at the end).

Row schema (same as the GSM8K / pirate trainers consume; extra ``meta`` keys are
ignored by the trainer):

    {"prompt": "Question: <request>\\nAnswer:",
     "completion": " ```python\\n<program>\\n```",
     "meta": {"kind": ..., "image": ..., "namespace": ..., "service_type": ...,
              "pvc": ..., "configmap": ..., "replicas": ..., "port": ...}}

Outputs ``runs/pulumi/train.jsonl`` (160 rows) and ``runs/pulumi/eval.jsonl``
(32 rows). Deterministic (seeded). The eval split is held out on purpose: three
images and two namespaces never appear in train, and every eval row's
(image, kind, service_type, has_pvc) tuple is unseen in train.

``--check-tokens`` additionally tokenizes every row with the Qwen2.5-0.5B
tokenizer (offline, from the repo venv) and asserts prompt + completion fit in
900 tokens.
"""

import argparse
import json
import os
import random
import shlex
import sys
import textwrap
from pathlib import Path

SEED = 2409
N_TRAIN = 160
N_EVAL = 32
MAX_TOKENS = 900
OUT_DIR = Path("runs/pulumi")
VENV_PYTHON = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"

# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

# image -> (role, default port, resource name, mount path for a data volume)
IMAGES = {
    "nginx:1.27": ("web", 80, "nginx", "/usr/share/nginx/html"),
    "httpd:2.4.62": ("web", 80, "httpd", "/usr/local/apache2/htdocs"),
    "caddy:2.8": ("web", 80, "caddy", "/data"),
    "ghcr.io/acme/api:v1.4.2": ("web", 8080, "api", "/var/lib/api"),
    "ghcr.io/acme/web:v2.0.1": ("web", 3000, "web", "/app/uploads"),
    "ghcr.io/acme/worker:v0.9.3": ("web", 9000, "worker", "/var/spool/worker"),
    "registry.example.com/shop/checkout:1.12.0": ("web", 8080, "checkout", "/var/lib/checkout"),
    "grafana/grafana:11.2.0": ("monitoring", 3000, "grafana", "/var/lib/grafana"),
    "prom/prometheus:v2.54.1": ("monitoring", 9090, "prometheus", "/prometheus"),
    "redis:7.2": ("data", 6379, "redis", "/data"),
    "postgres:16": ("data", 5432, "postgres", "/var/lib/postgresql/data"),
    "mysql:8.4": ("data", 3306, "mysql", "/var/lib/mysql"),
    "mongo:7.0": ("data", 27017, "mongo", "/data/db"),
    "rabbitmq:3.13": ("data", 5672, "rabbitmq", "/var/lib/rabbitmq"),
    "memcached:1.6.29": ("data", 11211, "memcached", "/data"),
    "nats:2.10": ("data", 4222, "nats", "/data"),
    "prom/node-exporter:v1.8.2": ("daemon", 9100, "node-exporter", None),
    "fluent/fluent-bit:3.1.8": ("daemon", 2020, "fluent-bit", None),
    "grafana/promtail:3.0.0": ("daemon", 9080, "promtail", None),
    "busybox:1.36": ("job", None, "busybox", None),
    "curlimages/curl:8.10.1": ("job", None, "curl", None),
    "python:3.12-slim": ("job", None, "python", None),
    "alpine:3.20": ("job", None, "alpine", None),
    "ghcr.io/acme/report:v0.5.0": ("job", None, "report", None),
}

# image -> (job name, command, mount path when the job gets a volume)
JOBS = {
    "busybox:1.36": ("cleanup", ["sh", "-c", "find /tmp -mtime +7 -delete"], "/work"),
    "curlimages/curl:8.10.1": ("healthcheck", ["curl", "-fsS", "http://api:8080/health"], "/work"),
    "python:3.12-slim": ("report", ["python", "-c", "import platform; print(platform.python_version())"], "/work"),
    "alpine:3.20": ("housekeeping", ["sh", "-c", "echo housekeeping done"], "/work"),
    "ghcr.io/acme/report:v0.5.0": ("daily-report", ["python", "report.py", "--daily"], "/work"),
    "postgres:16": ("pg-backup", ["sh", "-c", "pg_dump -h postgres -U app appdb > /backup/appdb.sql"], "/backup"),
    "mysql:8.4": ("mysql-backup", ["sh", "-c", "mysqldump -h mysql -u root appdb > /backup/appdb.sql"], "/backup"),
}

EVAL_ONLY_IMAGES = {"memcached:1.6.29", "mysql:8.4", "fluent/fluent-bit:3.1.8"}
NAMESPACES = ["default", "web", "data", "batch", "monitoring", "apps", "staging"]
EVAL_ONLY_NAMESPACES = {"staging", "monitoring"}

KIND_WEIGHTS = {
    "Deployment": 45,
    "StatefulSet": 15,
    "Job": 12,
    "CronJob": 12,
    "DaemonSet": 8,
    "DeploymentConfigMap": 4,
    "DeploymentPVC": 4,
}
KIND_ROLES = {
    "Deployment": {"web", "data", "monitoring"},
    "DeploymentConfigMap": {"web", "data", "monitoring"},
    "DeploymentPVC": {"web", "data", "monitoring"},
    "StatefulSet": {"data", "monitoring"},
    "DaemonSet": {"daemon"},
}

REPLICAS = [1, 2, 3, 5]
SERVICE_TYPES = ["ClusterIP", "ClusterIP", "NodePort", "LoadBalancer"]
PVC_SIZES = ["5Gi", "10Gi", "20Gi", "50Gi"]
BACKOFF_LIMITS = [2, 3, 4, 6]
SCHEDULES = {
    "*/5 * * * *": "every 5 minutes",
    "*/15 * * * *": "every 15 minutes",
    "0 * * * *": "every hour",
    "0 2 * * *": "daily at 02:00",
    "30 1 * * 0": "every Sunday at 01:30",
    "0 0 1 * *": "on the first of every month at midnight",
    "0 6 * * 1-5": "at 06:00 on weekdays",
}
CONFIG_VALUES = {
    "LOG_LEVEL": "info",
    "WORKERS": "4",
    "FEATURE_FLAGS": "beta",
    "CACHE_TTL": "300",
    "MAX_CONNECTIONS": "100",
    "TIMEOUT_SECONDS": "30",
    "REGION": "eu-west-1",
}
# tier -> (cpu limit, memory limit, cpu request, memory request)
TIERS = {
    "small": ("250m", "256Mi", "125m", "128Mi"),
    "medium": ("500m", "512Mi", "250m", "256Mi"),
    "large": ("1", "1Gi", "500m", "512Mi"),
}
DEFAULT_TIER = "medium"
P_LIMITS_SENTENCE = 0.25

# ---------------------------------------------------------------------------
# Natural-language request phrasing
# ---------------------------------------------------------------------------

LEADS = {
    "Deployment": [
        "Deploy {image}",
        "I need a Deployment running {image}",
        "Create Pulumi Python that runs {image} as a Deployment",
        "Write a pulumi_kubernetes program that deploys {image}",
        "Run {image} as a Kubernetes Deployment",
        "Please stand up {image}",
        "Using Pulumi Python, deploy {image}",
        "Set up a Deployment for {image}",
        "Spin up {image}",
        "Give me a Pulumi program that deploys {image}",
    ],
    "StatefulSet": [
        "Run {image} as a StatefulSet",
        "I need a StatefulSet running {image}",
        "Create Pulumi Python for a {image} StatefulSet",
        "Set up a stateful {image} deployment",
        "Write a pulumi_kubernetes StatefulSet for {image}",
        "Stand up {image} as a StatefulSet",
        "Deploy {image} as a StatefulSet",
    ],
    "DaemonSet": [
        "Run {image} on every node",
        "Deploy {image} as a DaemonSet",
        "I need {image} running on each node of the cluster",
        "Create a pulumi_kubernetes DaemonSet for {image}",
        "Write Pulumi Python that runs {image} on all nodes",
        "Put {image} on every node",
    ],
    "Job": [
        "Run a one-off Job",
        "Create a Kubernetes Job",
        "I need a batch Job",
        "Write Pulumi Python for a Job",
        "Set up a Job with Pulumi",
        "Define a pulumi_kubernetes Job",
    ],
    "CronJob": [
        "Set up a job on schedule `{schedule}`",
        "Create a CronJob running {when}",
        "I need a CronJob scheduled `{schedule}`",
        "Write Pulumi Python for a CronJob that runs {when}",
        "Schedule a job {when}",
        "Define a pulumi_kubernetes CronJob with schedule `{schedule}`",
    ],
}
LEADS["DeploymentConfigMap"] = LEADS["Deployment"]
LEADS["DeploymentPVC"] = LEADS["Deployment"]

REPLICA_CLAUSES = ["with {n} replicas", "scaled to {n} replicas", "running {n} replicas"]
SINGLE_REPLICA_CLAUSES = ["with 1 replica", "as a single replica", "with a single replica"]
PORT_CLAUSES = ["on port {port}", "listening on port {port}", "serving on port {port}"]
NS_CLAUSES = ["in namespace {ns}", "in the {ns} namespace", "into the {ns} namespace"]
SVC_CLAUSES = [
    "exposed as a {svc} service",
    "behind a {svc} Service",
    "fronted by a {svc} service",
    "exposed through a {svc} service",
]
HEADLESS_CLAUSES = ["with a headless service", "fronted by a headless service", "exposed via a headless Service"]
PVC_CLAUSES = [
    "with a {size} persistent volume mounted at {path}",
    "backed by a {size} PVC at {path}",
    "storing data on a {size} volume at {path}",
]
VCT_CLAUSES = [
    "with a {size} volume claim template at {path}",
    "where each replica gets a {size} persistent volume at {path}",
    "with per-pod {size} storage mounted at {path}",
]
JOB_PVC_CLAUSES = ["writing to a {size} volume mounted at {path}", "with a {size} PVC at {path}"]
CMD_CLAUSES = ["that runs `{cmd}` in {image}", "running `{cmd}` in {image}", "using {image} to run `{cmd}`"]
BACKOFF_CLAUSES = ["with a backoff limit of {b}", "retrying at most {b} times", "with backoff_limit {b}"]
CM_CLAUSES = [
    "reading {keys} from a ConfigMap",
    "with {keys} injected as env vars from a ConfigMap",
    "configured via ConfigMap keys {keys}",
]
CM_SENTENCES = [
    "Read {keys} from a ConfigMap.",
    "Inject {keys} as environment variables from a ConfigMap.",
    "It should pick up {keys} from a ConfigMap.",
]
LIMIT_SENTENCES = [
    "Limit it to {cpu} CPU and {mem} memory.",
    "Cap each container at {cpu} CPU and {mem} of memory.",
    "Set resource limits of {cpu} CPU / {mem} memory.",
]


def join_words(items):
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def sentence(lead, fixed, clauses):
    clauses = list(clauses)
    random.shuffle(clauses)
    parts = [lead] + fixed + clauses
    if len(parts) > 2:
        return " ".join(parts[:-1]) + ", " + parts[-1] + "."
    return " ".join(parts) + "."


def make_prompt(spec):
    kind = spec["kind"]
    image = spec["image"]
    lead = random.choice(LEADS[kind]).format(
        image=image, schedule=spec.get("schedule"), when=SCHEDULES.get(spec.get("schedule"))
    )
    ns = random.choice(NS_CLAUSES).format(ns=spec["namespace"])
    fixed, clauses, extra = [], [ns], []

    if kind in ("Deployment", "DeploymentConfigMap", "DeploymentPVC", "StatefulSet"):
        n = spec["replicas"]
        clauses.append(random.choice(SINGLE_REPLICA_CLAUSES if n == 1 else REPLICA_CLAUSES).format(n=n))
        clauses.append(random.choice(PORT_CLAUSES).format(port=spec["port"]))
        if kind == "StatefulSet":
            clauses.append(random.choice(HEADLESS_CLAUSES))
            clauses.append(random.choice(VCT_CLAUSES).format(size=spec["pvc"], path=spec["pvc_path"]))
        else:
            clauses.append(random.choice(SVC_CLAUSES).format(svc=spec["service_type"]))
        if kind == "DeploymentPVC":
            clauses.append(random.choice(PVC_CLAUSES).format(size=spec["pvc"], path=spec["pvc_path"]))
        if kind == "DeploymentConfigMap":
            keys = join_words(spec["configmap"])
            if random.random() < 0.5:
                clauses.append(random.choice(CM_CLAUSES).format(keys=keys))
            else:
                extra.append(random.choice(CM_SENTENCES).format(keys=keys))
    elif kind == "DaemonSet":
        clauses.append(random.choice(PORT_CLAUSES).format(port=spec["port"]))
    else:  # Job / CronJob
        fixed.append(random.choice(CMD_CLAUSES).format(cmd=shlex.join(spec["command"]), image=image))
        clauses.append(random.choice(BACKOFF_CLAUSES).format(b=spec["backoff_limit"]))
        if spec["pvc"]:
            clauses.append(random.choice(JOB_PVC_CLAUSES).format(size=spec["pvc"], path=spec["pvc_path"]))

    if spec["tier_mentioned"]:
        cpu, mem = TIERS[spec["tier"]][:2]
        extra.append(random.choice(LIMIT_SENTENCES).format(cpu=cpu, mem=mem))

    text = sentence(lead, fixed, clauses)
    if extra:
        text += " " + " ".join(extra)
    return text


# ---------------------------------------------------------------------------
# Spec sampling
# ---------------------------------------------------------------------------

def sample_spec(images, namespaces):
    kind = random.choices(list(KIND_WEIGHTS), weights=list(KIND_WEIGHTS.values()))[0]
    if kind in ("Job", "CronJob"):
        image = random.choice([i for i in images if i in JOBS])
    else:
        image = random.choice([i for i in images if IMAGES[i][0] in KIND_ROLES[kind]])
    role, port, name, mount = IMAGES[image]
    tier_mentioned = random.random() < P_LIMITS_SENTENCE
    spec = {
        "kind": kind,
        "image": image,
        "name": name,
        "namespace": random.choice(namespaces),
        "port": port,
        "replicas": None,
        "service_type": "none",
        "pvc": None,
        "pvc_path": None,
        "configmap": None,
        "command": None,
        "backoff_limit": None,
        "schedule": None,
        "tier": random.choice(list(TIERS)) if tier_mentioned else DEFAULT_TIER,
        "tier_mentioned": tier_mentioned,
    }
    if kind in ("Deployment", "DeploymentConfigMap", "DeploymentPVC"):
        spec["replicas"] = random.choice(REPLICAS)
        spec["service_type"] = random.choice(SERVICE_TYPES)
        if kind == "DeploymentPVC":
            spec["pvc"], spec["pvc_path"] = random.choice(PVC_SIZES), mount
        if kind == "DeploymentConfigMap":
            spec["configmap"] = random.sample(list(CONFIG_VALUES), k=random.choice([2, 3]))
    elif kind == "StatefulSet":
        spec["replicas"] = random.choice(REPLICAS)
        spec["service_type"] = "headless"
        spec["pvc"], spec["pvc_path"] = random.choice(PVC_SIZES), mount
    elif kind in ("Job", "CronJob"):
        job_name, command, job_mount = JOBS[image]
        spec["name"], spec["command"] = job_name, command
        spec["backoff_limit"] = random.choice(BACKOFF_LIMITS)
        if random.random() < 0.3:
            spec["pvc"], spec["pvc_path"] = random.choice(PVC_SIZES), job_mount
        if kind == "CronJob":
            spec["schedule"] = random.choice(list(SCHEDULES))
    return spec


# ---------------------------------------------------------------------------
# Program rendering
# ---------------------------------------------------------------------------

METADATA = "metadata=k8s.meta.v1.ObjectMetaArgs(name=name, namespace=namespace, labels=labels),"


def indent(text, n):
    return textwrap.indent(text, " " * n)


def render_container(spec):
    cpu_l, mem_l, cpu_r, mem_r = TIERS[spec["tier"]]
    lines = [
        "k8s.core.v1.ContainerArgs(",
        "    name=name,",
        f'    image="{spec["image"]}",',
    ]
    if spec["port"]:
        lines.append(f"    ports=[k8s.core.v1.ContainerPortArgs(container_port={spec['port']})],")
    if spec["command"]:
        lines.append(f"    command={json.dumps(spec['command'])},")
    if spec["configmap"]:
        lines += [
            "    env=[",
            "        k8s.core.v1.EnvVarArgs(",
            "            name=key,",
            "            value_from=k8s.core.v1.EnvVarSourceArgs(",
            "                config_map_key_ref=k8s.core.v1.ConfigMapKeySelectorArgs(name=config_name, key=key)",
            "            ),",
            "        )",
            "        for key in config_data",
            "    ],",
        ]
    if spec["pvc"]:
        lines.append(
            f'    volume_mounts=[k8s.core.v1.VolumeMountArgs(name="data", mount_path="{spec["pvc_path"]}")],'
        )
    lines += [
        "    security_context=k8s.core.v1.SecurityContextArgs(",
        "        run_as_non_root=True,",
        "        allow_privilege_escalation=False,",
        "    ),",
        "    resources=k8s.core.v1.ResourceRequirementsArgs(",
        f'        limits={{"cpu": "{cpu_l}", "memory": "{mem_l}"}},',
        f'        requests={{"cpu": "{cpu_r}", "memory": "{mem_r}"}},',
        "    ),",
        ")",
    ]
    return "\n".join(lines)


def render_pod_template(spec, restart_policy=None, volume_claim=None):
    lines = [
        "k8s.core.v1.PodTemplateSpecArgs(",
        "    metadata=k8s.meta.v1.ObjectMetaArgs(labels=labels),",
        "    spec=k8s.core.v1.PodSpecArgs(",
    ]
    if restart_policy:
        lines.append(f'        restart_policy="{restart_policy}",')
    lines += ["        containers=[", indent(render_container(spec), 12), "        ],"]
    if volume_claim:
        lines += [
            "        volumes=[",
            "            k8s.core.v1.VolumeArgs(",
            '                name="data",',
            f"                persistent_volume_claim=k8s.core.v1.PersistentVolumeClaimVolumeSourceArgs(claim_name={volume_claim}),",
            "            )",
            "        ],",
        ]
    lines += ["    ),", ")"]
    return "\n".join(lines)


def render_header(spec):
    lines = [
        "import pulumi",
        "import pulumi_kubernetes as k8s",
        "",
        f'name = "{spec["name"]}"',
        f'namespace = "{spec["namespace"]}"',
        'labels = {"app": name}',
    ]
    if spec["configmap"]:
        data = ", ".join(f'"{k}": "{CONFIG_VALUES[k]}"' for k in spec["configmap"])
        lines += ['config_name = f"{name}-config"', f"config_data = {{{data}}}"]
    return "\n".join(lines) + "\n\n"


def render_configmap():
    return "\n".join([
        "config = k8s.core.v1.ConfigMap(",
        "    config_name,",
        "    metadata=k8s.meta.v1.ObjectMetaArgs(name=config_name, namespace=namespace, labels=labels),",
        "    data=config_data,",
        ")",
        "",
        "",
    ])


def render_pvc(spec):
    return "\n".join([
        "pvc = k8s.core.v1.PersistentVolumeClaim(",
        '    f"{name}-data",',
        '    metadata=k8s.meta.v1.ObjectMetaArgs(name=f"{name}-data", namespace=namespace, labels=labels),',
        "    spec=k8s.core.v1.PersistentVolumeClaimSpecArgs(",
        '        access_modes=["ReadWriteOnce"],',
        f'        resources=k8s.core.v1.VolumeResourceRequirementsArgs(requests={{"storage": "{spec["pvc"]}"}}),',
        "    ),",
        ")",
        "",
        "",
    ])


def render_service(spec):
    port = spec["port"]
    if spec["service_type"] == "headless":
        type_line = '        cluster_ip="None",'
    else:
        type_line = f'        type="{spec["service_type"]}",'
    return "\n".join([
        "service = k8s.core.v1.Service(",
        "    name,",
        f"    {METADATA}",
        "    spec=k8s.core.v1.ServiceSpecArgs(",
        type_line,
        "        selector=labels,",
        f"        ports=[k8s.core.v1.ServicePortArgs(port={port}, target_port={port})],",
        "    ),",
        ")",
        "",
        "",
    ])


def render_deployment(spec):
    out = render_header(spec)
    if spec["configmap"]:
        out += render_configmap()
    if spec["pvc"]:
        out += render_pvc(spec)
    template = render_pod_template(spec, volume_claim="pvc.metadata.name" if spec["pvc"] else None)
    out += "\n".join([
        "deployment = k8s.apps.v1.Deployment(",
        "    name,",
        f"    {METADATA}",
        "    spec=k8s.apps.v1.DeploymentSpecArgs(",
        f"        replicas={spec['replicas']},",
        "        selector=k8s.meta.v1.LabelSelectorArgs(match_labels=labels),",
        "        template=" + indent(template, 8).lstrip() + ",",
        "    ),",
        ")",
        "",
        "",
    ])
    out += render_service(spec)
    out += 'pulumi.export("service_name", service.metadata.name)\n'
    return out


def render_statefulset(spec):
    out = render_header(spec)
    out += render_service(spec)
    template = render_pod_template(spec)
    out += "\n".join([
        "statefulset = k8s.apps.v1.StatefulSet(",
        "    name,",
        f"    {METADATA}",
        "    spec=k8s.apps.v1.StatefulSetSpecArgs(",
        "        service_name=name,",
        f"        replicas={spec['replicas']},",
        "        selector=k8s.meta.v1.LabelSelectorArgs(match_labels=labels),",
        "        template=" + indent(template, 8).lstrip() + ",",
        "        volume_claim_templates=[",
        "            k8s.core.v1.PersistentVolumeClaimArgs(",
        '                metadata=k8s.meta.v1.ObjectMetaArgs(name="data"),',
        "                spec=k8s.core.v1.PersistentVolumeClaimSpecArgs(",
        '                    access_modes=["ReadWriteOnce"],',
        f'                    resources=k8s.core.v1.VolumeResourceRequirementsArgs(requests={{"storage": "{spec["pvc"]}"}}),',
        "                ),",
        "            )",
        "        ],",
        "    ),",
        ")",
        "",
        "",
    ])
    out += 'pulumi.export("statefulset_name", statefulset.metadata.name)\n'
    return out


def render_daemonset(spec):
    out = render_header(spec)
    template = render_pod_template(spec)
    out += "\n".join([
        "daemonset = k8s.apps.v1.DaemonSet(",
        "    name,",
        f"    {METADATA}",
        "    spec=k8s.apps.v1.DaemonSetSpecArgs(",
        "        selector=k8s.meta.v1.LabelSelectorArgs(match_labels=labels),",
        "        template=" + indent(template, 8).lstrip() + ",",
        "    ),",
        ")",
        "",
        "",
    ])
    out += 'pulumi.export("daemonset_name", daemonset.metadata.name)\n'
    return out


def render_job_spec(spec, base_indent):
    template = render_pod_template(
        spec, restart_policy="Never", volume_claim="pvc.metadata.name" if spec["pvc"] else None
    )
    lines = [
        "k8s.batch.v1.JobSpecArgs(",
        f"    backoff_limit={spec['backoff_limit']},",
        "    template=" + indent(template, 4).lstrip() + ",",
        ")",
    ]
    return indent("\n".join(lines), base_indent).lstrip()


def render_job(spec):
    out = render_header(spec)
    if spec["pvc"]:
        out += render_pvc(spec)
    out += "\n".join([
        "job = k8s.batch.v1.Job(",
        "    name,",
        f"    {METADATA}",
        "    spec=" + render_job_spec(spec, 4) + ",",
        ")",
        "",
        "",
    ])
    out += 'pulumi.export("job_name", job.metadata.name)\n'
    return out


def render_cronjob(spec):
    out = render_header(spec)
    if spec["pvc"]:
        out += render_pvc(spec)
    out += "\n".join([
        "cronjob = k8s.batch.v1.CronJob(",
        "    name,",
        f"    {METADATA}",
        "    spec=k8s.batch.v1.CronJobSpecArgs(",
        f'        schedule="{spec["schedule"]}",',
        "        job_template=k8s.batch.v1.JobTemplateSpecArgs(",
        "            spec=" + render_job_spec(spec, 12) + ",",
        "        ),",
        "    ),",
        ")",
        "",
        "",
    ])
    out += 'pulumi.export("cronjob_name", cronjob.metadata.name)\n'
    return out


RENDERERS = {
    "Deployment": render_deployment,
    "DeploymentConfigMap": render_deployment,
    "DeploymentPVC": render_deployment,
    "StatefulSet": render_statefulset,
    "DaemonSet": render_daemonset,
    "Job": render_job,
    "CronJob": render_cronjob,
}


def render(spec):
    return RENDERERS[spec["kind"]](spec).rstrip("\n")


# ---------------------------------------------------------------------------
# Rows and splits
# ---------------------------------------------------------------------------

def base_kind(kind):
    return "Deployment" if kind.startswith("Deployment") else kind


def make_row(spec):
    return {
        "prompt": f"Question: {make_prompt(spec)}\nAnswer:",
        "completion": f" ```python\n{render(spec)}\n```",
        "meta": {
            "kind": base_kind(spec["kind"]),
            "image": spec["image"],
            "namespace": spec["namespace"],
            "service_type": spec["service_type"],
            "pvc": spec["pvc"],
            "configmap": spec["configmap"],
            "replicas": spec["replicas"],
            "port": spec["port"],
            "tier": spec["tier"],
            "schedule": spec["schedule"],
            "backoff_limit": spec["backoff_limit"],
        },
    }


def combo(spec):
    return (spec["image"], base_kind(spec["kind"]), spec["service_type"], bool(spec["pvc"]))


def build_splits():
    train_images = [i for i in IMAGES if i not in EVAL_ONLY_IMAGES]
    train_namespaces = [n for n in NAMESPACES if n not in EVAL_ONLY_NAMESPACES]

    train, seen_prompts, train_combos = [], set(), set()
    while len(train) < N_TRAIN:
        spec = sample_spec(train_images, train_namespaces)
        row = make_row(spec)
        if row["prompt"] in seen_prompts:
            continue
        seen_prompts.add(row["prompt"])
        train_combos.add(combo(spec))
        train.append(row)

    eval_, tries = [], 0
    while len(eval_) < N_EVAL:
        tries += 1
        assert tries < 100_000, "could not fill the eval split with unseen combinations"
        spec = sample_spec(list(IMAGES), NAMESPACES)
        if combo(spec) in train_combos:
            continue
        row = make_row(spec)
        if row["prompt"] in seen_prompts:
            continue
        seen_prompts.add(row["prompt"])
        eval_.append(row)

    train_prompts = {r["prompt"] for r in train}
    eval_prompts = {r["prompt"] for r in eval_}
    assert len(train_prompts) == len(train) and len(eval_prompts) == len(eval_)
    assert train_prompts.isdisjoint(eval_prompts)
    return train, eval_


def report(train, eval_):
    for split_name, split in (("train", train), ("eval", eval_)):
        counts = {}
        for r in split:
            counts[r["meta"]["kind"]] = counts.get(r["meta"]["kind"], 0) + 1
        print(f"{split_name}: {len(split)} rows " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    train_images = {r["meta"]["image"] for r in train}
    train_ns = {r["meta"]["namespace"] for r in train}
    unseen_image = sum(r["meta"]["image"] not in train_images for r in eval_)
    unseen_ns = sum(r["meta"]["namespace"] not in train_ns for r in eval_)
    print(f"eval rows with an unseen image: {unseen_image}/{len(eval_)}; unseen namespace: {unseen_ns}/{len(eval_)}")
    lines = [r["completion"].count("\n") - 1 for r in train + eval_]
    print(f"program length: min={min(lines)} max={max(lines)} mean={sum(lines) / len(lines):.1f} lines")


def write_splits(train, eval_):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split in (("train", train), ("eval", eval_)):
        path = OUT_DIR / f"{split_name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in split))
        print(f"{path}: {len(split)} rows")


def check_tokens(rows):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as exc:  # noqa: BLE001 - report and fail this flag only
        print(f"--check-tokens: cannot load {TOKENIZER} offline ({type(exc).__name__}: {exc}).")
        print(f"Run this script with {VENV_PYTHON} (the tokenizer is cached in the repo venv).")
        sys.exit(2)
    totals = []
    for r in rows:
        n = len(tok(r["prompt"]).input_ids) + len(tok(r["completion"]).input_ids)
        assert n <= MAX_TOKENS, f"row exceeds {MAX_TOKENS} tokens ({n}): {r['prompt'][:80]!r}"
        totals.append(n)
    print(f"tokens per row: max={max(totals)} mean={sum(totals) / len(totals):.1f} (limit {MAX_TOKENS})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-tokens", action="store_true", help="tokenize every row and enforce the budget")
    args = parser.parse_args()
    if args.check_tokens and VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_PYTHON.parent.parent.resolve():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

    random.seed(SEED)
    train, eval_ = build_splits()
    write_splits(train, eval_)
    report(train, eval_)
    if args.check_tokens:
        check_tokens(train + eval_)


if __name__ == "__main__":
    main()
