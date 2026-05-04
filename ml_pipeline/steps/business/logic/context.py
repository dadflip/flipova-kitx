from typing import Dict, Any

def validate_business_context(
    project_name: str,
    domain: str,
    problem: str,
    impact: str,
    latency_req: str,
    interpretability: bool,
    dyn_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Valide et structure le contexte métier et les paramètres spécifiques au domaine."""
    target = dyn_params.get("target") or dyn_params.get("ts_target") or dyn_params.get("node_target") or dyn_params.get("edge_target")
    
    return {
        "project_name": project_name,
        "domain": domain,
        "problem": problem,
        "impact": impact,
        "latency_req": latency_req,
        "interpretability": interpretability,
        "domain_parameters": dyn_params,
        "target": target
    }
