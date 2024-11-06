/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wordvectors;

import static common.CommonVariables.FIELD_WORD_NAME;
import static common.CommonVariables.FIELD_WORD_VEC;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.store.FSDirectory;


public class WordVecs {
    
    static Properties prop;
    static IndexReader wvecReader;
    static IndexSearcher wvecSearcher;
    static IndexReader clusterInfoReader;
    static int expansionTerm;

    /**
     * Word Vector hashmap.
     */
    static HashMap<String, WordVec> wvecMap = new HashMap();

    /**
     * Word Cluster hashmap.
     */
    static HashMap<String, Integer> clusterMap = new HashMap();
    
    static public void init(String propFile) throws Exception {

        prop = new Properties();
        prop.load(new FileReader(propFile));        
        String loadFrom = prop.getProperty("wvecs.index");
        File indexDir = new File(loadFrom);
        
        wvecReader = DirectoryReader.open(FSDirectory.open(indexDir.toPath()));
        wvecSearcher = new IndexSearcher(wvecReader);
        
        int numVocabClusters = Integer.parseInt(prop.getProperty("retrieve.vocabcluster.numclusters", "0"));
        if (numVocabClusters > 0) {
            String clusterInfoIndexPath = prop.getProperty("wvecs.clusterids.basedir") + "/" + numVocabClusters;
            clusterInfoReader = DirectoryReader.open(FSDirectory.open(new File(clusterInfoIndexPath).toPath()));
        }
        
        System.out.println("Loading cluster ids in memory...");
        int numDocs = clusterInfoReader.numDocs();
        for (int i=0; i<numDocs; i++) {
            Document d = clusterInfoReader.document(i);
            String wordName = d.get(FIELD_WORD_NAME);
            int clusterId = Integer.parseInt(d.get(FIELD_WORD_VEC));
            clusterMap.put(wordName, clusterId);
        }
        
        System.out.println("Loading wvecs in memory...");
        numDocs = wvecReader.numDocs();
        for (int i=0; i<numDocs; i++) {
            Document d = wvecReader.document(i);
            String wordName = d.get(FIELD_WORD_NAME);
            String line = d.get(FIELD_WORD_VEC);
            WordVec wv = new WordVec(line);
            wv.normalize();
            wvecMap.put(wordName, wv);
        }

        expansionTerm = Integer.parseInt(prop.getProperty("expansionTerm", "15"));
    }

    public void loadWordVecsInMem(String propFile) throws IOException {

        prop = new Properties();
        prop.load(new FileReader(propFile));        
        String loadFrom = prop.getProperty("wvecs.index");
        File indexDir = new File(loadFrom);

        wvecReader = DirectoryReader.open(FSDirectory.open(indexDir.toPath()));
        wvecSearcher = new IndexSearcher(wvecReader);
        
        System.out.println("Loading wvecs in memory...");
        int numDocs = wvecReader.numDocs();
        for (int i=0; i<numDocs; i++) {
            Document d = wvecReader.document(i);
            String wordName = d.get(FIELD_WORD_NAME);
            String line = d.get(FIELD_WORD_VEC);
            WordVec wv = new WordVec(line);
            wv.normalize();
            wvecMap.put(wordName, wv);
        }

    }

    static public void close() throws Exception {
        wvecReader.close();
    }

    /**
     * Returns a Lucene query object with 'word'
     * @param word
     * @return 
     */
    static Query getLuceneQueryObject(String word) {
        BooleanQuery q = new BooleanQuery();
        TermQuery tq = new TermQuery(new Term(FIELD_WORD_NAME, word));
        q.add(tq, BooleanClause.Occur.MUST);
        return q;
    }
    
    static public WordVec getVecCached(String word) throws Exception {
        return wvecMap.get(word);
    }
    
    static public WordVec getVec(String word) throws Exception {
        TopScoreDocCollector collector;
        TopDocs topDocs;
        collector = TopScoreDocCollector.create(1);
        wvecSearcher.search(getLuceneQueryObject(word), collector);
        topDocs = collector.topDocs();
        
        if (topDocs.scoreDocs == null || topDocs.scoreDocs.length == 0) {
            //System.err.println("vec for word: " + word + " not found");
            return null;
        }
        
        int wordId = topDocs.scoreDocs[0].doc;
        Document matchedWordVec = wvecReader.document(wordId);
        String line = matchedWordVec.get(FIELD_WORD_VEC);
        WordVec wv = new WordVec(line);
        return wv;
    }

    static public float getSim(String u, String v) throws Exception {
        WordVec uVec = getVec(u);
        WordVec vVec = getVec(v);
        return uVec.cosineSim(vVec);
    }
    
    static public float getSim(WordVec u, WordVec v) throws Exception {
        return u.cosineSim(v);
    }
    
    static public WordVec getCentroid(List<WordVec> wvecs) {
        int numVecs = wvecs.size(), j, dimension = wvecs.get(0).getDimension();
        WordVec centroid = new WordVec(dimension);
        for (WordVec wv : wvecs) {
            for (j = 0; j < dimension; j++) {
                centroid.vec[j] += wv.vec[j];
            }
        }
        for (j = 0; j < dimension; j++) {
            centroid.vec[j] /= (double)numVecs;
        }
        centroid.setClusterId(wvecs.get(0).getClusterId());
        return centroid;
    }

    /**
     * Returns the clusterId associated with @word
     * @param word
     * @return clusterId - Integer
     * @throws Exception 
     */
    static public int getClusterId(String word) throws Exception {
        if (!clusterMap.containsKey(word))
            return -1;
        return clusterMap.get(word);
    }

    /**
     * Compute list of similar terms of the WordVec w and return
     * @param w
     * @return 
     */
    public static List<WordVec> computeNNs(WordVec w) {
        ArrayList<WordVec> distList = new ArrayList<>(wvecMap.size());
        
        if (w == null)
            return null;
        
        for (Map.Entry<String, WordVec> entry : wvecMap.entrySet()) {
            WordVec wv = entry.getValue();
            if (wv.word.equals(w.word)) // ignoring computing similarity with itself
                continue;
            wv.querySim = w.cosineSim(wv);
            distList.add(wv);
        }
        Collections.sort(distList);
        return distList.subList(0, Math.min(expansionTerm, distList.size()));        
    }


    public static void main(String[] args) {
        try {
            WordVecs.init("init.properties");
            System.out.println(WordVecs.getVec("govern"));
            //System.out.println(WordVecs.computeNNs(getVec("govern")));
            WordVecs.close();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
