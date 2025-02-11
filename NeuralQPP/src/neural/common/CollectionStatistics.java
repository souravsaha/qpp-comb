package neural.common;

import static neural.common.CommonVariables.FIELD_BOW;
import static neural.common.CommonVariables.FIELD_FULL_BOW;
import static neural.common.CommonVariables.FIELD_ID;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;


public class CollectionStatistics {

    /**
     * Path of the directory in which the index is stored.
     */
    String          indexPath;

    /**
     * The IndexReader that will be used for reading the index.
     */
    IndexReader     indexReader;

    /**
     * Total number of documents in the collection.
     */
    private long    docCount;
    /**
     * Total number of terms in the collection.
     */
    private long    vocSize;
    /**
     * Field for which, the statistics will be made.
     */
    public String   field;
    /**
     * Total number of unique terms in the collection 'in that field'.
     * NOTE: This value is different from luke-Number Of Terms 
     *       Luke-Number Of Terms = number of terms in all fields taken together.
     */
    private int     uniqTermCount;

    /**
     * perTerm statistics of all the terms of collection.
     */
    public HashMap<String, PerTermStat> perTermStat;

    public long getDocCount() {return docCount;}
    public long getVocSize() {return vocSize;}
    public int getUniqueTermCount() {return uniqTermCount;}

    /**
     * Constructor
     * @param indexPath Path of the index
     * @param field The field of the index which will be searched
     * @throws IOException 
     */
    public CollectionStatistics(String indexPath, String field) throws IOException {
        indexReader = DirectoryReader.open(FSDirectory.open(new File(indexPath).toPath()));
        this.field = field;
        perTermStat = new HashMap<>();
    }

    public CollectionStatistics(String indexPath) throws IOException {
        indexReader = DirectoryReader.open(FSDirectory.open(new File(indexPath).toPath()));
        perTermStat = new HashMap<>();
    }

    /**
     * Default constructor.
     */
    public CollectionStatistics() {
        perTermStat = new HashMap<>();
    }

    /**
     * Returns the vocabulary size of the index for 'field'.
     * @param indexReader
     * @param field
     * @return Total number of terms in 'field' of the index
     * @throws IOException 
     */
    public long getVocabularySize(IndexReader indexReader, String field) throws IOException {

        Fields fields = MultiFields.getFields(indexReader);
        Terms terms = fields.terms(field);
        if(null == terms) {
            System.err.println("Field: "+field);
            System.err.println("Error buildCollectionStat(): terms Null found");
        }
        long vocSize = terms.getSumTotalTermFreq();  // total number of terms in the index in that field

        return vocSize;
    }

    /**
     * Initialize collectionStat:<p>
     * docCount      - total-number-of-docs-in-index<p>
     * vocSize       - collection-size<p>
     * uniqTermCount - unique terms in collection<p>
     * perTermStat   - cf, df of each terms in the collection <p>
     * @throws IOException 
     */
    public void buildCollectionStat() throws IOException {

        docCount = indexReader.maxDoc();      // total number of documents in the index

        Fields fields = MultiFields.getFields(indexReader);
        Terms terms = fields.terms(field);
        if(null == terms) {
            System.err.println("Field: "+field);
            System.err.println("Error buildCollectionStat(): terms Null found");
        }
        vocSize = terms.getSumTotalTermFreq();  // total number of terms in the index in that field
        vocSize = getVocabularySize(indexReader, field);
        TermsEnum iterator = terms.iterator();
        BytesRef byteRef = null;

        while((byteRef = iterator.next()) != null) {
        //* for each word in the collection
            String t = new String(byteRef.bytes, byteRef.offset, byteRef.length);
            int df = iterator.docFreq();           // df of 't'
            long cf = iterator.totalTermFreq();    // cf of 't'
            // idf = log(#docCount / (df+1) )
            double idf = Math.log((float)(docCount)/(float)(df+1));
            double norm_cf = (double)cf / (double)vocSize;
            perTermStat.put(t, new PerTermStat(t, cf, df, idf, norm_cf));
        }
        uniqTermCount = perTermStat.size();

        System.out.println("Collection statistics built");
        System.out.println("Unique terms: " + uniqTermCount);
    }

    public void setUniqueTermCount() throws IOException {

        docCount = indexReader.maxDoc();      // total number of documents in the index

        Fields fields = MultiFields.getFields(indexReader);
        Terms terms = fields.terms(field);
        if(null == terms) {
            System.err.println("Field: "+field);
            System.err.println("Error buildCollectionStat(): terms Null found");
        }
        vocSize = terms.getSumTotalTermFreq();  // total number of terms in the index in that field
        vocSize = getVocabularySize(indexReader, field);
        TermsEnum iterator = terms.iterator();
        BytesRef byteRef = null;

        int count = 0;
        while((byteRef = iterator.next()) != null) {
        //* for each word in the collection
            String t = new String(byteRef.bytes, byteRef.offset, byteRef.length);
            count++;
        }
        uniqTermCount = count;

        System.out.println("Unique terms: " + uniqTermCount);
    }

    public double getIdf(String term, IndexReader indexReader, String fieldName) throws IOException {
        Fields fields = MultiFields.getFields(indexReader);
        Term termInstance = new Term(fieldName, term);
        long df = indexReader.docFreq(termInstance);       // DF: Returns the number of documents containing the term

        double idf = Math.log((float)(docCount)/(float)(df+1));

        return idf;
    }
    /**
     * Prints the basic statistics of the collection on the screen:<p>
     * Collection size, #documents in collection, #unique terms in collection,
     * Individual term df, cf in the collection.
     */
    public void showCollectionStat() {

        System.out.println("Collection Size: " + vocSize);
        System.out.println("Number of documents in collection: " + docCount);
        System.out.println("NUmber of unique terms in collection: " + uniqTermCount);

        ///*
        for (Map.Entry<String, PerTermStat> entrySet : perTermStat.entrySet()) {
            String key = entrySet.getKey();
            PerTermStat value = entrySet.getValue();
            System.out.println("Term: <"+key + "> " + 
                "df: "+value.getDF() +" cf: "+value.getCF());
        }
        //*/
    }

    public void getQueryDocumentVector(int luceneDocid, String[] qTerms, float lambda) throws IOException {

        DocumentVector dv = new DocumentVector();
        if(this == null)
            System.out.println("CollectionStatistics is null");
        dv = dv.getDocumentVector(luceneDocid, this);

        // Term vector for this document and field, or null if term vectors were not indexed
        Terms terms = indexReader.getTermVector(luceneDocid, FIELD_BOW);
        if(null == terms) {
            System.err.println("Error: getQueryDocumentVector() Term vectors not indexed: "+luceneDocid);
            System.exit(1);
        }

        float totalScore = 0;
        for(String qTerm : qTerms) {
            PerTermStat perQueryStat = dv.docPerTermStat.get(qTerm);
            //System.out.println(qTerm);
            if(null != perQueryStat) {
                long tf = perQueryStat.getCF();
                long docSize = dv.getDocSize();
                long cf = perTermStat.get(qTerm).getCF();
                long collSize = getVocSize();
                double singleTermScore = Math.log(1+((1-lambda)*tf/docSize)/(lambda * cf/collSize));
                totalScore += singleTermScore;
                System.out.print(qTerm+" TF: "+tf+" DocLen: "+docSize+" CF: "+cf+" CollSize: "+collSize);
                System.out.println(" Score: "+singleTermScore);
            }
//        (float)Math.log(1 + ((1 - lambda) * freq / docLen) /(lambda * ((LMStats)stats).getCollectionProbability()));
        }
        System.out.println("Total score: " + totalScore+"\n");
    }
    /**
     * Displays the vector of the document with luceneDocId
     * @param luceneDocid The lucene doc id of the corresponding document
     * @param indexReader The index reader
     * @throws IOException 
     */
    public void showDocumentVector(int luceneDocid, IndexReader indexReader) throws IOException {

        int docSize = 0;

        if(indexReader==null) {
            System.out.println("Error: null == indexReader in showDocumentVector(int,IndexReader)");
            System.exit(1);
        }

        // Term vector for this document and field, or null if term vectors were not indexed
        Terms terms = indexReader.getTermVector(luceneDocid, FIELD_BOW);
        if(null == terms) {
            System.err.println("Error: Term vectors not indexed: "+luceneDocid);
            System.exit(1);
        }

        System.out.println("Unique term count: " + terms.size());
        TermsEnum iterator = terms.iterator();
        BytesRef byteRef = null;

        while((byteRef = iterator.next()) != null) {
        //* for each word in the document
            String term = new String(byteRef.bytes, byteRef.offset, byteRef.length);
            long docFreq = perTermStat.get(term).getDF();            // df of 't'
            long colFreq = perTermStat.get(term).getCF();            // cf of 't'
            long termFreq = iterator.totalTermFreq();    // tf of 't'
            System.out.println(term+": tf: "+termFreq + " df: "+docFreq
                + " cf: " + colFreq);
            docSize += termFreq;
        }
        System.out.println("DocSize: "+docSize);
    }

    /**
     * Prints all the document ids with 1) full-length and 2) clean-length.
     * @throws IOException 
     */
    public void allDocLength() throws IOException {

        for (int i=0; i<indexReader.maxDoc(); i++) {
            Document doc = indexReader.document(i);
            String docId = doc.get(FIELD_ID);

            System.out.print((i+1) + " " + docId + " ");

            int fullLen, cleanLen;
            TermsEnum iterator;
            BytesRef byteRef;

            // Full length calculation
            // Term vector for this document and field, or null if term vectors were not indexed
            Terms terms = indexReader.getTermVector(i, FIELD_FULL_BOW);
            if(null == terms) {
                fullLen = 0;
            }

            else {
                iterator = terms.iterator();
                byteRef = null;

                fullLen = 0;
                while((byteRef = iterator.next()) != null) {
                //* for each word in the document
                    long termFreq = iterator.totalTermFreq();    // tf of 't'
                    fullLen += termFreq;
                }
            }

            // Clean length calculation
            // Term vector for this document and field, or null if term vectors were not indexed
            terms = indexReader.getTermVector(i, FIELD_BOW);
            if(null == terms) {
                cleanLen = 0;
            }
            else {
                iterator = terms.iterator();
                byteRef = null;

                cleanLen = 0;
                while((byteRef = iterator.next()) != null) {
                //* for each word in the document
                    long termFreq = iterator.totalTermFreq();    // tf of 't'
                    cleanLen += termFreq;
                }
            }
            System.out.println(fullLen + " " + cleanLen);
        }
    }

    public static void main(String[] args) throws IOException {

        String indexPath;
        String field;
        CollectionStatistics cs;

        if (args.length < 2) {
            System.out.println("Usage: java common.CollectionStatistics <index-path> [-d (to compute all doc.length)] "
                + "[-t <field-name> (to get the unique term count in field-name)]");
            System.exit(0);
        }
        indexPath = args[0];

        for(int i=1; i<args.length; i++) {
            String ch = args[i];
            switch(ch) {
                case "-d":
                    System.out.println("Computing all the document length");
                    cs = new CollectionStatistics(indexPath);
                    cs.allDocLength();
                    break;
                case "-t":
                    field = args[++i];
                    System.out.println("Computing unique term count in field: " + field);
                    cs = new CollectionStatistics(indexPath, field);
                    cs.setUniqueTermCount();
                    break;
                default:
                    cs = new CollectionStatistics();
                    break;
            }
        }       

//        cs.buildCollectionStat();
        //cs.showCollectionStat();

        //* show the document vector of the document with 1th lucene-docid
//        cs.showDocumentVector(1, cs.indexReader);
//        DocumentVector dv = new DocumentVector();
//        dv = dv.getDocumentVector(1, cs);
//        dv.printDocumentVector();
//        
    }
}

