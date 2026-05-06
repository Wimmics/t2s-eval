from t2s_soa.papers import construct_paper_json
from t2s_soa.verify_used_ref import (
    check_accepted_paper_used_metrics,
    check_metrics_ref_are_accepted,
    extract_metrics,
)

if __name__ == "__main__":
    construct_paper_json()
    extract_metrics()
    # check_bibtex_json()
    # check_bibtex_metrics_json()

    check_accepted_paper_used_metrics()
    check_metrics_ref_are_accepted()
