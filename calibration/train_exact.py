"""Training entry point for FUTURE strict exact-prefix, deploy-policy rows.

This never flips deployment_validated/publication_ready: online numerical parity
and independent evaluation must be audited before a release asset is assembled.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k,"2")
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import numpy as np
from .contracts import (ContractError, asset_metadata_errors, validate_collection_row,
    canonical_qid, split_question, ASSET_SCHEMA, CLASSES, sha256_file, hot_map_digest)
from .candidates import select_scored_candidate, FrozenHotVocabulary, CandidateGenerator
from .model import LEGACY_PAIR_POLICY, features, fit, predict, metrics


def write_json(path, data):
    with path.open("x", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
        out.write("\n")


def question_weights(qids):
    counts = Counter(qids)
    return np.asarray([1.0 / counts[q] for q in qids])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rows", type=Path,required=True)
    p.add_argument("--contract",type=Path,required=True,help="fully bound runtime metadata JSON")
    p.add_argument("--hot-map",type=Path,required=True,help="JSON list of target token IDs")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--evaluate-keep-native1", action="store_true",
                   help="Report classifier as an A-preserving B-to-X gate rather than posterior Top-2")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    contract=json.loads(a.contract.read_text(encoding="utf-8"))
    errors=asset_metadata_errors(contract,contract)
    if errors: raise ContractError("invalid collection contract: "+"; ".join(errors))
    hot=json.loads(a.hot_map.read_text(encoding="utf-8"))
    # Same ordered LE-int64 digest as ContextRepair; file bytes are a separate receipt.
    hot_map_file_sha256=sha256_file(a.hot_map)
    hot_map_ordered_digest=hot_map_digest(hot)
    if hot_map_ordered_digest!=contract["hot_map_sha256"]:
        raise ContractError("hot-map ordered LE-int64 digest mismatch")
    if len(hot)!=contract["draft_vocab_size"] or len(set(hot))!=len(hot):
        raise ContractError("hot-map not a unique full draft-to-target map")
    hot=FrozenHotVocabulary(hot,contract["target_vocab_size"])
    generator=CandidateGenerator(hot,capacity=contract["capacity"],policy=contract["candidate_policy"])
    rows=[];excluded=Counter();contents={};question_content={}
    opener=gzip.open if a.rows.suffix==".gz" else open
    with opener(a.rows,"rt",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            try: validate_collection_row(r,contract,hot,candidate_generator=generator)
            except (ContractError,ValueError) as e:
                excluded[str(e)]+=1;continue
            q=canonical_qid(r); c=r["question_content_sha256"]
            if "split" in r and r["split"] != split_question(q):
                raise ContractError("prepared data split disagrees with frozen canonical question split")
            if c in contents and contents[c]!=q:
                raise ContractError("same question content has different canonical IDs; canonicalize before splitting")
            if q in question_content and question_content[q]!=c:
                raise ContractError("canonical question ID maps to multiple content hashes")
            contents[c]=q;question_content[q]=c;rows.append(r)
    write_json(a.out/"exclusions.json",dict(excluded))
    if not rows: raise ContractError("no strict usable rows; collect new exact-prefix data")
    xs=[];ys=[];qids=[];sources=[]
    for r in rows:
        token,score=select_scored_candidate(r["candidate_ids"],r["candidate_scores"])
        xs.append(features(r["p1"],r["p2"],r["native2_logit"],score if score is not None else 0,
                           r["depth"],r["has_x"]))
        ys.append(0 if r["gold"]==r["native1"] else 1 if r["gold"]==r["native2"] else
                  2 if token is not None and r["gold"]==token else 3)
        qids.append(canonical_qid(r));sources.append(r["source"])
    x,y,qids,sources=np.asarray(xs),np.asarray(ys),np.asarray(qids),np.asarray(sources)
    split=np.asarray([split_question(q) for q in qids]);train=split=="train";hx=x[:,-1].astype(bool)
    coverage={"states":len(rows),"questions":len(set(qids)),"no_x_states":int((~hx).sum()),
              "train_no_x_states":int((~hx&train).sum()),"parents":dict(Counter(r["parent_kind"] for r in rows)),
              "train_parents":dict(Counter(r["parent_kind"] for r,t in zip(rows,train) if t)),
              "train_target_events":{name:int((y[train]==i).sum()) for i,name in enumerate(CLASSES)},
              "split_questions":{s:len(set(qids[split==s])) for s in ("train","validation","test")}}
    write_json(a.out/"coverage.json",coverage)
    if any(not coverage["split_questions"][s] for s in coverage["split_questions"]):
        raise ContractError("empty question split")
    missing_parents = [s for s in ("root", "native1", "native2", "x") if not coverage["train_parents"].get(s)]
    if missing_parents or (not a.evaluate_keep_native1 and not coverage["train_no_x_states"]):
        raise ContractError("missing training support for the requested scoring policy")
    if a.evaluate_keep_native1 and any(not coverage["train_target_events"][s] for s in ("native2", "x")):
        raise ContractError("B-to-X gate training needs both successful swaps and examples where keeping B is correct")
    params,info=fit(x[train],y[train],hx[train],sample_weight=question_weights(qids[train]))
    if not info["converged"]:raise ContractError("optimizer did not converge; refusing asset")
    pair_policy=LEGACY_PAIR_POLICY if a.evaluate_keep_native1 else contract["candidate_pair_policy"]
    report={"fit":info,"splits":{},"source_holdouts":[],"deployment_validated":False,
            "evaluated_candidate_pair_policy":pair_policy}
    for s in ("train","validation","test"):
        m=split==s
        report["splits"][s]=metrics(predict(params,x[m],hx[m]),y[m],qids[m],hx[m],
                                   pair_policy=pair_policy)
    # Hold out entire sources, also excluding source-alias question IDs from training.
    for source in sorted(set(sources)):
        test=sources==source; testq=set(qids[test]); tr=train&~np.isin(qids,list(testq))
        if not tr.any():continue
        pp,ff=fit(x[tr],y[tr],hx[tr],sample_weight=question_weights(qids[tr]))
        report["source_holdouts"].append({"source":source,"fit":ff,
           "metrics":metrics(predict(pp,x[test],hx[test]),y[test],qids[test],hx[test],
                             pair_policy=pair_policy)})
    asset=dict(schema=ASSET_SCHEMA,model_name="exact_prefix_candidate_pending_validation",
        deployment_validated=False,publication_ready=False,classes=list(CLASSES),metadata=contract,
        parameters=params,validation_evidence={"exact_prefix_labels":False,"per_row_parity":False,
        "no_x_observed":bool(coverage["train_no_x_states"]),"root_parent_observed":True,"native1_parent_observed":True,"native2_parent_observed":True,
        "x_parent_observed":True,"deployment_readout_parity":False,"independent_question_evaluation":False},
        collection_attestations={"matching_parent_target_prefix_hashes":True,
            "actual_target_forward_declared":True,"native_readout_parity_declared":True,
            "target_execution_receipts_independently_checked":False},
        training={"dataset_sha256":sha256_file(a.rows),"training_question_ids":sorted(set(qids[train])),
                  "hot_map_json_file_sha256":hot_map_file_sha256,
                  "hot_map_ordered_le_int64_sha256":hot_map_ordered_digest,
                  "fit":info,"split_seed":"context-repair-20261114-v1"},
        limitations=["Row parity attestations do not establish online fused/Graph equivalence.",
          "Matching prefix hashes plus execution IDs are declarations, not independently checked Target receipts.",
          "Question holdout from one collected archive is not an independent blind data collection.",
          "Must confirm alternate-prefix labels came from actual Target execution, not another greedy suffix."])
    write_json(a.out/"candidate_pending_validation.json",asset)
    write_json(a.out/"evaluation.json",report)
    print(json.dumps(coverage,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
