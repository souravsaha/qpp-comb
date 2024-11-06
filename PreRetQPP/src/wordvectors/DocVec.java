/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wordvectors;

import static common.CommonVariables.FIELD_WORD_VEC;
//import indexer.CompressionUtils;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.highlight.QueryTermExtractor;
import org.apache.lucene.search.highlight.WeightedTerm;


public class DocVec {
    
    String          id;     // doc id
    String          text;   // Ensure that you pass a preprocessed list of tokens delimited by white spaces.
    HashMap<String, WordVec> wvecMap;
    KMeansPlusPlusClusterer<WordVec> clusterer;
    
    static final float EPSILON = .001f; // for DBSCAN clustering 
    static final int MIN_PTS = 1;       // for DBSCAN clustering 
    
    public DocVec(String id, String text) throws Exception {
        this.id = id;
        this.text = text;
        wvecMap = new HashMap<>();
        // clusterer = new DBSCANClusterer<>(EPSILON, MIN_PTS);
        buildTerms();
    }

    /**
     * Constructor, only called during retrieval;
     * Purpose is to construct vector representation of query terms.
     * @param query The query
     * @throws Exception 
     */
    public DocVec(Query query) throws Exception {

        WeightedTerm[] qterms = QueryTermExtractor.getTerms(query);
        wvecMap = new HashMap<>();

        for (WeightedTerm wterm: qterms) {
            WordVec qwv = WordVecs.getVec(wterm.getTerm());
            if (qwv != null) {
                qwv.normalize();
                wvecMap.put(qwv.getWord(), qwv);
            }
        }
    }
    
    // This constructor is going to be invoked during similarity
    // computation in the retrieval phase, i.e. we would have to
    // load the clustered representation (stored in the index)
    // to build up the wordmap (we won't need the words nor would
    // we have them, because these would be cluster centre points rather
    // than individual word points.
    /**
     * This constructor is going to be invoked during similarity
     * computation in the retrieval phase, i.e. we would have to
     * load the clustered representation (stored in the index)
     * to build up the wordmap (we won't need the words nor would
     * we have them, because these would be cluster centre points rather
     * than individual word points.
     * @param doc
     * @param compressedIndex 
     */
    public DocVec(Document doc, boolean compressedIndex) {
        int numClusterCentres;
        wvecMap = new HashMap<>();
        String clusterCentres;
        
        // If we are going to use per-doc clusters
//        if (!compressedIndex)
            clusterCentres = doc.get(FIELD_WORD_VEC);
//        else
//            clusterCentres = CompressionUtils.decompress(doc.getBinaryValue(FIELD_WORD_VEC).bytes);

        String[] clusterCentreVecStrings = clusterCentres.split(":");
        numClusterCentres = clusterCentreVecStrings.length;
        for (int i = 0; i < numClusterCentres; i++) {
            WordVec wv = new WordVec(clusterCentreVecStrings[i]);
            wvecMap.put(wv.word, wv);
        }
    }
    
    // Directly pass a map set from else where
    public DocVec(HashMap<String, WordVec> wvecMap) {
        this.wvecMap = wvecMap;
    }
    
    void buildTerms() throws Exception {
        String[] tokens = text.split("\\s+");
        for (String token: tokens) {
            WordVec wv = wvecMap.get(token);
            if (wv == null) {
                wv = WordVecs.getVec(token);
                if (wv != null)
                    wvecMap.put(token, wv);
            }
        }
    }

    /**
     * Clustering using KMeans.
     * @param numClusters The number of clusters to make
     * @return A List of Cluster(WordVec)
     * @throws Exception 
     */
    public List<CentroidCluster<WordVec>> clusterWords(int numClusters) throws Exception {
        System.out.println("Clustering document: " + this.id);
        List<WordVec> wordList = new ArrayList<>(wvecMap.size());
        for (Map.Entry<String, WordVec> e : wvecMap.entrySet()) {
            wordList.add(e.getValue());
        }
        
        clusterer = new KMeansPlusPlusClusterer<>(Math.min(numClusters, wordList.size()));
        List<CentroidCluster<WordVec>> clusters = clusterer.cluster(wordList);        
        return clusters;
    }

    /**
     * Cluster using DBScan.
     * @return A List of Cluster(WordVec)
     */
    public List<Cluster<WordVec>> clusterWords() {
        System.out.println("Clustering document: " + this.id);
        List<WordVec> wordList = new ArrayList<>(wvecMap.size());
        for (Map.Entry<String, WordVec> e : wvecMap.entrySet()) {
            wordList.add(e.getValue());
        }
        
        DBSCANClusterer dbscan = new DBSCANClusterer(EPSILON, MIN_PTS);
        List<Cluster<WordVec>> clusters = dbscan.cluster(wordList);
        return clusters;
    }
    
    public String getClusterVecs(int numClusters) throws Exception {
        StringBuffer buff = new StringBuffer();
        List<CentroidCluster<WordVec>> clusters = clusterWords(numClusters);
        int i = 0;
        
        for (CentroidCluster<WordVec> c : clusters) {
            //List<WordVec> thisClusterPoints = c.getPoints();
            //WordVec clusterCenter = WordVecs.getCentroid(thisClusterPoints);
            Clusterable clusterCenter = c.getCenter();
            WordVec clusterWordVec = new WordVec("Cluster_" + i, clusterCenter.getPoint());
            //clusterCenter.setWord("Cluster_" + numClusters);
            buff.append(clusterWordVec.toString()).append(":");
            i++;
        }
        
        return buff.toString();
    }

    /**
     * Returns a list of terms (with vector) of the document.
     * @return List containing WordVec of terms of the document
     */
    public List<WordVec> getWordVecs() {
        List<WordVec> wvList = new ArrayList<>();
        for (Map.Entry<String, WordVec> e : wvecMap.entrySet()) {
            wvList.add(e.getValue());
        }
        return wvList;
    }

    // For unit testing
    public static void main(String[] args) {
        List<DoublePoint> plist = new ArrayList<>();
        int[][] points = {{1, 1}, {0, 0}, {1, 0}, {0, 1},
                     {10, 10}, {9, 9}, {10, 9}, {9, 10}};
        int K = 3;
        KMeansPlusPlusClusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(K);
        
        for (int[] point: points) {
            DoublePoint dp = new DoublePoint(point);
            plist.add(dp);
            System.out.println(dp);
        }
        System.out.println("Clustering with K-Means. K = "+K);
        int clusterId = 0;
        List<CentroidCluster<DoublePoint>> pointCluster = clusterer.cluster(plist);
        for (CentroidCluster<DoublePoint> db : pointCluster) {
            List<DoublePoint> clusterPoints = db.getPoints();
            for (DoublePoint clusterPoint : clusterPoints) {
                System.out.println(clusterId + " : " + Arrays.toString(clusterPoint.getPoint()));
            }
            clusterId++;
            System.out.println();
        }
    }
}
