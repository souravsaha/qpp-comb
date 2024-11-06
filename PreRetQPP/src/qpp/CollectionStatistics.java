/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package qpp;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;



public class CollectionStatistics {
    String      indexPath;
    IndexReader indexReader;
    String      field;
    int         docCount;       // total number of documents in the collection
    long        colSize;        // total number of terms in the collection
    int         uniqTermCount;  // total number of unique terms in the collection
    /* NOTE: This value is different from luke-Number Of Terms 
        I think luke-Number Of Terms = docCount+uniqTermCount.
        It is same in this case
    */
    HashMap<String, PerTermStat> perTermStat;

    public CollectionStatistics(String indexPath, String field) throws IOException {
        indexReader = DirectoryReader.open(FSDirectory.open(new File(indexPath).toPath()));
        this.field = field;
        perTermStat = new HashMap<>();
    }

    public CollectionStatistics() {
        perTermStat = new HashMap<>();
    }

    /**
     * Initialize collectionStat:
     * docCount      - total-number-of-docs-in-index
     * colSize       - collection-size
     * uniqTermCount - unique terms in collection
     * perTermStat   - cf, df of each terms in the collection 
     * @throws IOException 
     */
    public void buildCollectionStat() throws IOException {

        docCount = indexReader.maxDoc();      // total number of documents in the index

        Fields fields = MultiFields.getFields(indexReader);
        Terms terms = fields.terms(field);
        colSize = terms.getSumTotalTermFreq();  // total number of terms in the index in that field
        TermsEnum iterator = terms.iterator();
        BytesRef byteRef = null;

        while((byteRef = iterator.next()) != null) {
        //* for each word in the collection
            String term = new String(byteRef.bytes, byteRef.offset, byteRef.length);
            int docFreq = iterator.docFreq();           // df of 'term'
            long colFreq = iterator.totalTermFreq();    // cf of 'term'
            double idf = 1+Math.log(docCount/docFreq);
            double rcf = (double)colFreq / (double)colSize;
            perTermStat.put(term, new PerTermStat(term, colFreq, docFreq, idf, rcf));
        }
        uniqTermCount = perTermStat.size();

        System.out.println("Collection statistics built");
//        return collectionStat;
    }

    public void showCollectionStat() {
        System.out.println("Collection Size: " + colSize);
        System.out.println("Number of documents in collection: " + docCount);
        System.out.println("NUmber of unique terms in collection: " + uniqTermCount);

//        /*
        for (Map.Entry<String, PerTermStat> entrySet : perTermStat.entrySet()) {
            String key = entrySet.getKey();
            PerTermStat value = entrySet.getValue();
            System.out.println("Term: <"+key + "> " + 
                "df: "+value.df +" cf: "+value.cf);
        }
  //      */
    }

    public void showDocumentVector(int luceneDocId, IndexReader indexReader) throws IOException {

        HashMap<String, Integer> docVec = new HashMap<>();
        int docSize = 0;

        if(indexReader==null) {
            System.out.println("Error: null == indexReader in showDocumentVector(int,IndexReader)");
            System.exit(1);
        }

        // term vector for this document and field, or null if term vectors were not indexed
        Terms terms = indexReader.getTermVector(luceneDocId, "content");
        if(null == terms) {
            System.err.println("Error: Term vectors not indexed");
            System.exit(1);
        }

        System.out.println(terms.size());
        TermsEnum iterator = terms.iterator();
        BytesRef byteRef = null;

        while((byteRef = iterator.next()) != null) {
        //* for each word in the document
            String term = new String(byteRef.bytes, byteRef.offset, byteRef.length);
            //int docFreq = iterator.docFreq();            // df of 'term'
            long termFreq = iterator.totalTermFreq();    // tf of 'term'
            System.out.println(term+": tf: "+termFreq);
            //collectionStat.perTermStat.put(term, new PerTermStat(term, colFreq, docFreq));
            docSize += termFreq;
        }
        System.out.println("DocSize: "+docSize);
    }

    public static void main(String[] args) throws IOException {
        String indexPath = "trec678.lucene";
        String field = "content";

        CollectionStatistics cs = new CollectionStatistics(indexPath, field);

        cs.buildCollectionStat();
        cs.showCollectionStat();
        //* show the document vector of the document with 1th lucene-docid
        //cs.showDocumentVector(1, cs.indexReader);
        DocumentVector dv = new DocumentVector();
        dv = dv.getDocumentVector(1, cs);
        dv.printDocumentVector();
    }
}

