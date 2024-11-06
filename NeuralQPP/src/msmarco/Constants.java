package msmarco;

public interface Constants {
    String ID_FIELD = "id";
    String CONTENT_FIELD = "words";
    String MSMARCO_COLL = "/home/suchana/.ir_datasets/msmarco-passage/collection.tsv";
    String MSMARCO_INDEX = "/store/index/msmarco-passages-lucene-index/index_lucene5/";
    //String MSMARCO_INDEX = "procheta_index/";
    String QRELS_TRAIN = "/home/suchana/PycharmProjects/ColBERT/data/trecDL_data/trecDL_q97.qrel";
    String QRELS_TRAIN_DEV_SMALL = "/home/suchana/.ir_datasets/msmarco-passage/dev/small/qrels";
    String QUERIES_TRAIN_DEV_SMALL = "/home/suchana/.ir_datasets/msmarco-passage/dev/small/queries.tsv";
    String QRELS_TRAIN_DEV = "data/qrels.dev.tsv";
    String QUERY_FILE_TRAIN = "/home/suchana/PycharmProjects/ColBERT/data/trecDL_data/trecDL_q97.tsv";
    String STOP_FILE = "/home/suchana/smart-stopwords";
    String QUERY_FILE_TEST = "data/trecdl/pass_2019.queries";
    String RES_FILE = "/home/suchana/Desktop/res.txt";
    String SAVED_MODEL = "/home/suchana/Desktop/model.tsv";
    int NUM_WANTED = 1000;
    final float LAMBDA = 0.4f;
}