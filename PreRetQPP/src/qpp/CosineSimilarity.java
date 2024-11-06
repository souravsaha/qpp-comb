/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package qpp;


import java.io.File;
import java.io.IOException;
import java.util.*;

import org.apache.commons.math3.linear.*;
import org.apache.lucene.index.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.*;

public class CosineSimilarity {

    public String SEARCH_FIELD = "words";
    IndexReader reader;

    public Set<String> terms = new HashSet<>();
    private RealVector v1;
    private RealVector v2;

    CosineSimilarity(IndexReader reader) {
        this.reader = reader;
    }

    CosineSimilarity(String indexPath, String searchField) throws IOException {
        Directory indexDir = FSDirectory.open(new File(indexPath).toPath());
        reader = DirectoryReader.open(indexDir);
        SEARCH_FIELD = searchField;
    }
    
    CosineSimilarity(IndexReader reader, String searchField) {
        this.reader = reader;
        SEARCH_FIELD = searchField;
    }
    
    CosineSimilarity(String indexPath) throws IOException {
        Directory indexDir = FSDirectory.open(new File(indexPath).toPath());
        reader = DirectoryReader.open(indexDir);
    }

    public double computeCosineSimilarity(int docid1, int docid2) throws IOException {
        Map<String, Integer> f1 = getTermFrequencies(reader, docid1);
        Map<String, Integer> f2 = getTermFrequencies(reader, docid2);

        v1 = toRealVector(f1);
        v2 = toRealVector(f2);
        return getCosineSimilarity();
    }

    double getCosineSimilarityValue() {
        return (v1.dotProduct(v2)) / (v1.getNorm() * v2.getNorm());
    }

    public double getCosineSimilarity() throws IOException {
        return getCosineSimilarityValue();
    }

    Map<String, Integer> getTermFrequencies(IndexReader reader, int docId) throws IOException {
        Terms vector = reader.getTermVector(docId, SEARCH_FIELD);
        TermsEnum termsEnum = null;
        termsEnum = vector.iterator();
        Map<String, Integer> frequencies = new HashMap<>();
        BytesRef text = null;
        while ((text = termsEnum.next()) != null) {
            String term = text.utf8ToString();
            int freq = (int) termsEnum.totalTermFreq();
            frequencies.put(term, freq);
            terms.add(term);
        }
        return frequencies;
    }

    RealVector toRealVector(Map<String, Integer> map) {
        RealVector vector = new ArrayRealVector(terms.size());
        int i = 0;
        for (String term : terms) {
            int value = map.containsKey(term) ? map.get(term) : 0;
            vector.setEntry(i++, value);
        }
        return (RealVector) vector.mapDivide(vector.getL1Norm());
    }

    public static void main(String[] args) throws IOException {
        CosineSimilarity cs = new CosineSimilarity("/store/collections/indexed/trec678-store-refined/");

        System.out.println(cs.computeCosineSimilarity(0, 1));
    }
}