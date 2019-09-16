import os
from pathlib import Path

SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PRO_ROOT = SRC_ROOT.parent

UTEST_ROOT = PRO_ROOT / "utest"
DATA_ROOT = PRO_ROOT / "data"
PDATA_ROOT = DATA_ROOT / "processed"
RESULT_PATH = PRO_ROOT / "results"

WHOLE_WIKI_DUMP_PATH = DATA_ROOT / "enwiki-20171001-pages-meta-current-withlinks-processed"
ABS_WIKI_DUMP_PATH = DATA_ROOT / "enwiki-20171001-pages-meta-current-withlinks-abstracts"

WHOLE_WIKI_FILE = PDATA_ROOT / "whole_file.jsonl"
WHOLE_WIKI_DB = PDATA_ROOT / "whole_file.db"
ABS_WIKI_FILE = PDATA_ROOT / "whole_abs_file.jsonl"
ABS_WIKI_DB = PDATA_ROOT / "whole_abs_file.db"
WIKI_TITLE_SET_FILE = PDATA_ROOT / "title_set.txt"

ABS_PROCESS_FOR_RINDEX_DB = PDATA_ROOT / "reverse_indexing" / "abs_for_rindex.db"
WHOLE_PROCESS_FOR_RINDEX_DB = PDATA_ROOT / "reverse_indexing" / "whole_for_rindex.db"
WHOLE_WIKI_RAW_TEXT = PDATA_ROOT / "wiki_raw_text.db"

DEV_FULLWIKI_FILE = DATA_ROOT / "hotpotqa" / "hotpot_dev_fullwiki_v1.json"
TEST_FULLWIKI_FILE = DATA_ROOT / "hotpotqa" / "hotpot_test_fullwiki_v1.json"
DEV_DISTRACTOR_FILE = DATA_ROOT / "hotpotqa" / "hotpot_dev_distractor_v1.json"
TRAIN_FILE = DATA_ROOT / "hotpotqa" / "hotpot_train_v1.1.json"

FEVER_DATA_ROOT = DATA_ROOT / "fever"
FEVER_TRAIN = FEVER_DATA_ROOT / "fever_1.0" / "train.jsonl"
FEVER_DEV = FEVER_DATA_ROOT / "fever_1.0" / "dev.jsonl"
FEVER_TEST = FEVER_DATA_ROOT / "fever_1.0" / "test.jsonl"

FEVER_TOKENIZED_DOC_ID = FEVER_DATA_ROOT / "tokenized_doc_id.json"

FEVER_DB = FEVER_DATA_ROOT / "fever.db"

SQUAD_DEV_1_1 = DATA_ROOT / "squad" / "dataset/dev-v1.1.json"
SQUAD_TRAIN_1_1 = DATA_ROOT / "squad" / "dataset/train-v1.1.json"
SQUAD_DEV_2_0 = DATA_ROOT / "squad" / "dataset/dev-v2.0.json"
SQUAD_TRAIN_2_0 = DATA_ROOT / "squad" / "dataset/train-v2.0.json"

CURATEDTREC_TRAIN = DATA_ROOT / "CuratedTrec/CuratedTrec-train.txt"
CURATEDTREC_TEST = DATA_ROOT / "CuratedTrec/CuratedTrec-test.txt"


OPEN_SQUAD_DEV_GT = PRO_ROOT / "data/p_squad/m_p_gt/squad_dev_p_gt.jsonl"
OPEN_SQUAD_TRAIN_GT = PRO_ROOT / "data/p_squad/m_p_gt/squad_train_p_gt.jsonl"

OPEN_WEBQ_TEST_GT = PRO_ROOT / "data/p_webq/m_p_gt/webq_test_p_gt.jsonl"
OPEN_WEBQ_TRAIN_GT = PRO_ROOT / "data/p_webq/m_p_gt/webq_train_p_gt.jsonl"

OPEN_WIKIM_TEST_GT = PRO_ROOT / "data/p_wikimovie/m_p_gt/wikimovie_test_p_gt.jsonl"
OPEN_WIKIM_TRAIN_GT = PRO_ROOT / "data/p_wikimovie/m_p_gt/wikimovie_train_p_gt.jsonl"

OPEN_CURATEDTERC_TEST_GT = PRO_ROOT / "data/p_curatedtrec/m_p_gt/curatedtrec_test_p_gt.jsonl"
OPEN_CURATEDTERC_TRAIN_GT = PRO_ROOT / "data/p_curatedtrec/m_p_gt/curatedtrec_train_p_gt.jsonl"


if __name__ == '__main__':
    print("PRO_ROOT", PRO_ROOT)
    print("SRC_ROOT", SRC_ROOT)
    print("UTEST_ROOT", UTEST_ROOT)
    print("DATA_ROOT", DATA_ROOT)

    print("WHOLE_WIKI_DUMP_PATH", WHOLE_WIKI_DUMP_PATH)