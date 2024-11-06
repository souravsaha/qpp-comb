/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package qpp;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;


public class DocumentVector {
    HashMap<String, PerTermStat>    docVec;
    int                             size;

    public DocumentVector() {
        docVec = new HashMap<>();
    }

    public DocumentVector(HashMap<String, PerTermStat> docVec, int size) {
        this.docVec = docVec;
        this.size = size;
    }

    /**
     * Returns the document vector for a document with lucene-docid=luceneDocId
     * Returns dv containing 
     *      1) docVec: a HashMap of (term,PerTermStat) type
     *      2) size : size of the document
     * @param luceneDocId
     * @param cs
     * @return
     * @throws IOException 
     */
    public DocumentVector getDocumentVector(int luceneDocId, CollectionStatistics cs) throws IOException {

        DocumentVector dv = new DocumentVector();
        int docSize = 0;

        if(cs.indexReader==null) {
            System.out.println("Error: null == indexReader in showDocumentVector(int,IndexReader)");
            System.exit(1);
        }

        // term vector for this document and field, or null if term vectors were not indexed
        Terms terms = cs.indexReader.getTermVector(luceneDocId, "content");
        if(null == terms) {
            System.err.println("Error: Term vectors not indexed");
            System.exit(1);
        }

        TermsEnum iterator = terms.iterator();
        BytesRef byteRef = null;

        //* for each word in the document
        while((byteRef = iterator.next()) != null) {
            String term = new String(byteRef.bytes, byteRef.offset, byteRef.length);
            //int docFreq = iterator.docFreq();            // df of 'term'
            long termFreq = iterator.totalTermFreq();    // tf of 'term'
            //System.out.println(term+": tf: "+termFreq);
            docSize += termFreq;

            //* termFreq = cf, in a document; df = 1, in a document
            //dv.docVec.put(term, new PerTermStat(term, termFreq, 1));
            dv.docVec.put(term, new PerTermStat(term, termFreq, 1, cs.perTermStat.get(term).idf, (double)termFreq/(double)cs.colSize));
        }
        dv.size = docSize;
        //System.out.println("DocSize: "+docSize);

        return dv;
    }

    /**
     * Returns true if the docVec is not-zero; else, return false.
     * @return 
     */
    public boolean printDocumentVector() {

        if(this == null) {
            System.err.println("Error: printing document vector. Calling docVec null");
            System.exit(1);
        }
        if(0 == docVec.size()) {
            System.out.println("Error: printing document vector. Calling docVec zero");
            return false;
        }
        for (Map.Entry<String, PerTermStat> entrySet : docVec.entrySet()) {
            String key = entrySet.getKey();
            PerTermStat value = entrySet.getValue();
            System.out.println(key + " : " + value.cf);
        }
        return true;
    }

    public long returnTf(String term, DocumentVector dv) {
        PerTermStat t = dv.docVec.get(term);
        if(null != t)
            return t.cf;
        else
            return 0;
    }
}
