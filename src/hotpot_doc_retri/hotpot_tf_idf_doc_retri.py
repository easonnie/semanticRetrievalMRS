from build_rindex.build_rvindex import IndexDB, load_from_file
from build_rindex.rvindex_scoring import get_top_ranked_tf_idf_doc
from evaluation import ext_hotpot_eval
from wiki_util.title_entities_set import get_title_entity_set
from utils import common
import config
from tqdm import tqdm


def toy_init_results():
    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    print(len(dev_fullwiki_list))

    # Load rindex file
    abs_rindexdb = IndexDB()
    abs_rindexdb.load_from_file(config.PDATA_ROOT / "reverse_indexing/abs_rindexdb")
    print("Number of terms:", len(abs_rindexdb.inverted_index.index))
    abs_rindexdb.inverted_index.build_Nt_table()
    abs_rindexdb.score_db['default-tf-idf'] = dict()
    load_from_file(abs_rindexdb.score_db['default-tf-idf'],
                   config.PDATA_ROOT / "reverse_indexing/abs_rindexdb/scored_db/default-tf-idf.score.txt")
    # Load rindex finished

    saved_items = []
    for item in tqdm(dev_fullwiki_list):
        saved_tfidf_item = dict()
        question = item['question']
        qid = item['_id']

        doc_list = get_top_ranked_tf_idf_doc(question, abs_rindexdb, top_k=50)
        saved_tfidf_item['question'] = question
        saved_tfidf_item['qid'] = qid
        saved_tfidf_item['doc_list'] = doc_list

        saved_items.append(saved_tfidf_item)

    common.save_jsonl(saved_items, config.RESULT_PATH / "doc_retri_results/term_based_methods_results/hotpot_tf_idf_dev.jsonl")


def load_and_eval():
    top_k = 50
    value_thrsehold = None
    tf_idf_dev_results = common.load_jsonl(config.RESULT_PATH / "doc_retri_results/term_based_methods_results/hotpot_tf_idf_dev.jsonl")
    doc_pred_dict = {'sp_doc': dict()}

    for item in tqdm(tf_idf_dev_results):
        sorted_scored_list = sorted(item['doc_list'], key=lambda x: x[0], reverse=True)
        pred_list = [docid for _, docid in sorted_scored_list[:top_k]]
        # print(sorted_scored_list)

        qid = item['qid']
        doc_pred_dict['sp_doc'][qid] = pred_list

        # break

    dev_fullwiki_list = common.load_json(config.DEV_FULLWIKI_FILE)
    ext_hotpot_eval.eval(doc_pred_dict, dev_fullwiki_list)


if __name__ == '__main__':
    # toy_init_results()
    load_and_eval()