from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

from ..data import SOURCES, _fetch, file_sha256

FILES = {
    "gandalf": {
        "train": (
            "data/train-00000-of-00001-ded53be747ff55cd.parquet",
            "5b6acf3e5a5998d21f8e1222bb45bbdec25a14408747b1cd63bebef4a75fa439",
        ),
        "validation": (
            "data/validation-00000-of-00001-94481a2a09ff2fff.parquet",
            "f51ab3e3407a368845b0f57932cc745c09280429416d7507bd178a16326a79f6",
        ),
        "test": (
            "data/test-00000-of-00001-bc92128b9288a6d1.parquet",
            "56b646d133335ebc535266bd55dbe1b5bee7caa4b95bf49d040684b9b5dd9972",
        ),
    },
    "llmail": {
        "phase1_labels": (
            "data/labelled_unique_submissions_phase1.json",
            "691dfa1595d2bd0e731069f233bd5448f7af1c0ebd9732bc6f12dd6cee446586",
        ),
        "phase2_labels": (
            "data/labelled_unique_submissions_phase2.json",
            "f89af984e345430c3b357903890e30867bf4676f4ef10c138cc7bad218e890b8",
        ),
        "phase1_raw": (
            "data/raw_submissions_phase1.jsonl",
            "a9c62eca699dd270fdfbbfbfcc1253f5e5017f6d5a34ff7cb0f2cbb80b7f7c0a",
        ),
        "phase2_raw": (
            "data/raw_submissions_phase2.jsonl",
            "a9207e1d893ccb74ca6f9cc5eecea433bc49c23a26bed88088afd385c7ab18b6",
        ),
        "false_positive_controls": (
            "data/emails_for_fp_tests.json",
            "4ddd950b5dbaa8548f5597c886d8e09a051ba07f80a9291fdcca9c2397d22abe",
        ),
    },
    "tensor_trust_raw": {
        "attacks_v2": (
            "raw-data/v2/raw_dump_attacks.jsonl.bz2",
            "87853cc8065a22156d15c4bdec777a8d35758749beb79769147d63a9644a73ce",
        ),
        "defenses_v2": (
            "raw-data/v2/raw_dump_defenses.jsonl.bz2",
            "cbdf52c469b2a57db61f885562bae1725cd56174cc201979e7bc042a680a5166",
        ),
        "extraction_detection_v1": (
            "detecting-extractions/v1/prompt_extraction_detection.jsonl",
            "af28d09554db4f8ed91d042005c3457f72c88e763e3787c3b1c76c1d9e8f8260",
        ),
    },
    "browsesafe": {
        "train": (
            "train.parquet",
            "430881fa53da3898956048676f58db894fc760cfa348ab264ee8a07c44a3d9fb",
        ),
        "test": (
            "test.parquet",
            "00cbad96b60fee46e016d79af6981fb221384c61f12cf28b4f04b5a6420573d0",
        ),
    },
    "hackaprompt": {
        "full": (
            "hackaprompt.parquet",
            "bedca308fbd71be57793930e4e4a0dcfbda2a27b6d0f2ad3191bb20a6a315928",
        ),
    },
    "wildjailbreak": {
        "train": (
            "train/train.tsv",
            "376719bfdb46ad1a19e7ba4f587f80cc7cb1368cc213ee647ef739c170550f7a",
        ),
        "eval": (
            "eval/eval.tsv",
            "eeb7e43aafa0151588f5cb8994b99adc0ee34a57819d92f43a6084f0b7fe4fa4",
        ),
    },
    "wildguardmix": {
        "train": (
            "train/wildguard_train.parquet",
            "02ecea8a724a9146a1e473a95a7cdf262adfe9c7d5408953ca86d2fcfbdc8953",
        ),
        "test": (
            "test/wildguard_test.parquet",
            "6ccc2909c1ae6d41424fac69f1fc32535b1de39cb8f80407d81e8bc64a0bebca",
        ),
    },
    "mind2web": {
        "0000": (
            "default/partial-train/0000.parquet",
            "38af0159c74d9c2ef242f593748368d78a4095c29f65391ba30f36082262e12e",
        ),
        "0001": (
            "default/partial-train/0001.parquet",
            "c8454678aee7f1ffda24524f10d30ac5d463ec8448ccbbcabd41603c4b851a38",
        ),
        "0002": (
            "default/partial-train/0002.parquet",
            "29a7dcf58b4e058166d48c8fbf568b4986e7a78fe89d1828c7e341e887c7f3f4",
        ),
        "0003": (
            "default/partial-train/0003.parquet",
            "e586dafff7bebf88ea08f1559b0fe7894b7e5429a983abb1dd4dc646a3e9cdcd",
        ),
        "0004": (
            "default/partial-train/0004.parquet",
            "c0825ada5cbace870feaad5a64977a3c5bee4a65683240aafa3b872fa413d71e",
        ),
        "0005": (
            "default/partial-train/0005.parquet",
            "7b64a352f86af77ed3357f35bfa2dc1cf227927d97e04017019a70f90fcb305d",
        ),
        "0006": (
            "default/partial-train/0006.parquet",
            "ef75bcd3e708018b3582a44b08f61c90a3d96740eec8a31dbe8f473b5c782dfd",
        ),
        "0007": (
            "default/partial-train/0007.parquet",
            "717b40e632290cefa2c0dd60dc304a702f1196002349bbd1a1689b555d233bb8",
        ),
    },
    "swebench_verified": {
        "dev_test": (
            "data/test-00000-of-00001.parquet",
            "43ed5a3d1d98da36472c1ade65ddd2085d7b4ff694fcaf6a023a07c5c1f32f21",
        ),
    },
    "false_reject": {
        "train": (
            "train.jsonl",
            "0331899da03e9c2c232acffd8b086e5e57116e1f53986012743b9a3bea46f868",
        ),
        "test": (
            "test.jsonl",
            "644b243987b4d16f36b1b668b03a17fa49d18174fee80fbaeef94a53facc462d",
        ),
    },
    "coconot": {
        "train": (
            "pref/train-00000-of-00001.parquet",
            "136ac18a54fbfa98472eabb77369de89c556ea05626584445f381238c287e104",
        ),
        "dev_test": (
            "contrast/test-00000-of-00001.parquet",
            "d2d4f9ea33eac017cfdd2b56669e417e979933418276083d3acb28be170a588f",
        ),
    },
    "jbb_benign": {
        "dev_test": (
            "data/benign-behaviors.csv",
            "3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468",
        ),
    },
    "lmsys_arena": {
        "train": (
            "data/train-00000-of-00001-cced8514c7ed782a.parquet",
            "3726a6352e9bfc34e206460646f6e5e99bb837751966a671ddd30c7f64e5b06e",
        ),
    },
    "agentic_boundary_pairs": {
        "train": (
            "data/train.jsonl",
            "98fc7ea70a45215ba50d21efcd79cfed33afcbc1ac95d0fc388066b42cea2238",
        ),
        "validation": (
            "data/validation.jsonl",
            "1e5f483277ca776f0bb7d26c1a85ca4cd6d73f8d30f638c75265d82a6de751c3",
        ),
        "test": (
            "data/test.jsonl",
            "034523eefaec18c291f9daa7699037acb0e7a91b1aeb1b205878b124f6e18f10",
        ),
    },
}


def _download(
    source: str,
    filename: str,
    expected_sha256: str,
    *,
    revision: str | None = None,
) -> tuple[Path, str]:
    info = SOURCES[source]
    try:
        path = Path(
            hf_hub_download(
                info["repo"],
                filename,
                repo_type="dataset",
                revision=revision or info["revision"],
            )
        )
    except GatedRepoError as error:
        if info.get("gated"):
            raise RuntimeError(
                f"{source}: access gate not accepted or token unavailable"
            ) from error
        raise
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ValueError(f"{source}:{filename} does not match its pinned digest")
    return path, digest


def _parquet_dataset(source: str) -> tuple[dict, dict[str, str]]:
    datasets = {}
    downloads = {}
    for split, (filename, expected) in FILES[source].items():
        path, digest = _download(source, filename, expected)
        downloads[filename] = digest
        datasets[split] = load_dataset(
            "parquet", data_files={split: str(path)}, split=split
        )
    return datasets, downloads


def _verified_archive(source: str) -> tuple[bytes, str]:
    info = SOURCES[source]
    return _fetch(
        info["archive_url"],
        max_bytes=info["bytes"],
        expected_bytes=info["bytes"],
        expected_sha256=info["sha256"],
    )
