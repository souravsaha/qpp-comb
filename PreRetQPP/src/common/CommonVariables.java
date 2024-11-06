
package common;

public class CommonVariables {

    /**
     * The unique document id of each of the documents.
     */
//    static final public String FIELD_ID = "docid";
    static final public String FIELD_ID = "id";

    /**
     * The analyzed content of each of the documents.
     */
//    static final public String FIELD_BOW = "content";
    static final public String FIELD_BOW = "words";

    /**
     * Analyzed full content (including tag informations): Mainly used for WT10G initial retrieval.
     */
    static final public String FIELD_FULL_BOW = "full-content";

    /* Following variables are used for the vector related information indexing */

    /**
     * The word of the vocabulary.
     */
    static final public String FIELD_WORD_NAME = "wordname";

    /**
     * The vector corresponding to a word of the vocabulary.
     */
    static final public String FIELD_WORD_VEC = "wordvec";

    /**
     * The cluster-id of word.
     */
    static final public String FIELD_WORD_CLUSTER = "wordcluster";

    /**
     * 
     */
    static final public String FIELD_PID = "paraId";
}
