/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package qpp;

import common.InputOutput;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Properties;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import common.TRECQueryParser;
import common.TRECQuery;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.search.similarities.DefaultSimilarity;
import org.xml.sax.SAXException;
import wordvectors.WordVec;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import static qpp.NQC.returnNQC;
import wordvectors.Silhouette;



public class Approach2 {

    Properties prop;
    String propPath;
    IndexReader vecReader;
    IndexReader clusterInfoReader;
    IndexSearcher vecSearcher;
    IndexSearcher clusterInfoSearcher;

    Analyzer analyzer;
    List<TRECQuery> queries;
    int numVocabClusters;
    TRECQueryParser trecQueryparser;
    String fieldToSearch;
    String wordClusterId, wordVector;
    HashMap<String, List<TermSimilarity>> termNNHash;
    String resPath;
    private String mapPath;
    double alpha;
    int noOfTopDocs;
    int desirableClusterSize;

    public Approach2(String propPath) throws IOException, SAXException, Exception {
        
        this.propPath = propPath;
        prop = new Properties();
        try {
            prop.load(new FileReader(propPath));
        } catch (IOException ex) {
            System.err.println("Error: Properties file missing in "+propPath);
            System.exit(1);
        }
        //----- Properties file loaded

        //+++++ index path setting 
        String indexPath;
        File indexFile;
        Directory indexDir;

        // +++ vector index
        indexPath = prop.getProperty("wvecs.indexPath");
        indexFile = new File(indexPath);
        indexDir = FSDirectory.open(indexFile.toPath());

        if (!DirectoryReader.indexExists(indexDir)) {
            System.err.println("Index doesn't exists in "+indexPath);
            System.exit(1);
        }
        /* setting reader and searcher */
        vecReader = DirectoryReader.open(FSDirectory.open(indexFile.toPath()));
        vecSearcher = new IndexSearcher(vecReader);
        vecSearcher.setSimilarity(new DefaultSimilarity());
        // --- vector index

        // +++ cluster index
//        numVocabClusters = Integer.parseInt(prop.getProperty("wvecs.vocabCluster.numClusters", "100"));
//        if (numVocabClusters > 0) {
//            String clusterInfoIndexPath = prop.getProperty("wvecs.vocabCluster.baseIndexPath") + "/" + numVocabClusters;
//            clusterInfoReader = DirectoryReader.open(FSDirectory.open(new File(clusterInfoIndexPath).toPath()));
//            clusterInfoSearcher = new IndexSearcher(clusterInfoReader);
//            clusterInfoSearcher.setSimilarity(new DefaultSimilarity());
//        }                
        // --- cluster index

        // +++
        termNNHash = new HashMap<>();
        String nnPath = prop.getProperty("wvecs.nnPath");
        readNNFile(nnPath, 30);
        // ---
        //----- index path set

        // +++++ setting the analyzer with English Analyzer with Smart stopword list
        String stopFilePath = prop.getProperty("stopFilePath");
        common.EnglishAnalyzerWithSmartStopword engAnalyzer = new common.EnglishAnalyzerWithSmartStopword(stopFilePath);
        analyzer = engAnalyzer.setAndGetEnglishAnalyzerWithSmartStopword();
        // ----- analyzer set: analyzer

        /* setting query path */
        String queryPath = prop.getProperty("queryPath");
        /* constructing the query */
        fieldToSearch = "wordname";
        trecQueryparser = new TRECQueryParser(queryPath, analyzer, fieldToSearch);
        queries = trecQueryparser.constructQueries();
        /* constructed the query */

        wordClusterId = "wordvec";
        wordVector = "wordvec";

        resPath = prop.getProperty("resPath");
        mapPath = prop.getProperty("mapPath");

        // linear smoothing parameter to weight NQC and QPP_wvec
        alpha = Double.parseDouble(prop.getProperty("alpha", "0.6"));
        System.out.println("Alpha set to "+alpha);

        desirableClusterSize = Integer.parseInt(prop.getProperty("desirableClusterSize", "6"));
        System.out.println("desirableClusterSize set to "+desirableClusterSize);
        // no of top docs to be used for NQC
        noOfTopDocs = Integer.parseInt(prop.getProperty("noOfTopDocs", "100"));
    }

    private void readNNFile(String nnPath, int nnCount) throws FileNotFoundException, IOException {

        FileInputStream fis = new FileInputStream(new File(nnPath));
        String rootTerm_NNWithSim[];        // root term, its NN terms along with similarity with root term
        List<TermSimilarity> nnList;
 
	//Construct BufferedReader from InputStreamReader
	BufferedReader br = new BufferedReader(new InputStreamReader(fis));
 
	String line = null;
        line = br.readLine();   // read first line containing <no.of.word> [SPACE] <no.of.NNs>
        int noOfWords = Integer.parseInt(line.split(" ")[0]);
        int noOfNNs = Integer.parseInt(line.split(" ")[1]);

        //System.out.println(noOfWords + " " + noOfNNs);

        while ((line = br.readLine()) != null) {
            rootTerm_NNWithSim = new String[noOfNNs];
            rootTerm_NNWithSim = line.split(";");  // noOfNNs+1 fields; first: root-term; then noOfNNs no. of terms with similarity
            nnList = new ArrayList<>();
            String rootWord = rootTerm_NNWithSim[0];
            //System.out.println("Root word: " + rootWord + " " + rootTerm_NNWithSim.length);
            for (int i = 0; i < Math.min(noOfNNs, nnCount)-1 ; i++) {
                String[] singleNNWithSim = rootTerm_NNWithSim[i+1].split(" ");
                nnList.add(new TermSimilarity(singleNNWithSim[0], Float.parseFloat(singleNNWithSim[1])));
            }
            termNNHash.put(rootWord, nnList);
	}
 
	br.close();

        System.out.println("NNs read into memory");
    }

    private void processAll() throws Exception {

        String queryTerms[];
        TopScoreDocCollector collector;
        ScoreDoc[] hits = null;
        TopDocs topDocs;
        Term t;
        Query q;
        QueryParser queryParser = new QueryParser(fieldToSearch, new WhitespaceAnalyzer());

        List<WordVec> listNNTerms;          // list of NN terms for a particular query; to be used for clustering
        HashSet<WordVec> hashNNTerms;   // hashmap of NN terms for a particular query; to be used for making the list for clustering
        int localNumClusters;

        double nqc, qpp;
        double allAP[], allQPP[];
        int processedQueryCount = 0;
        allAP = new double [queries.size()];
        allQPP = new double [queries.size()];

        HashMap<String, Double[]> queryScoreHash = InputOutput.readScoreFromResFile(this.resPath);
        HashMap<String, Double> queryMapHash = InputOutput.readAPFromFile(this.mapPath);

        for (TRECQuery query : queries) {

            listNNTerms = new ArrayList<>();
            hashNNTerms = new HashSet<>();

            Query luceneQuery = trecQueryparser.getAnalyzedQuery(query);
//            System.out.println(query.qid+ " " + luceneQuery.toString(fieldToSearch));
            if(!luceneQuery.toString(fieldToSearch).isEmpty()) {
                queryTerms = new String[luceneQuery.toString(fieldToSearch).split(" ").length];
                queryTerms = luceneQuery.toString(fieldToSearch).split(" ");

                Double[] qScore = queryScoreHash.get(query.qid);
                Double queryAP = queryMapHash.get(query.qid);

                if (null != qScore && null != queryAP) {
                    double score[] = ArrayUtils.toPrimitive(qScore);

                    nqc = returnNQC(score, noOfTopDocs);
    //                System.out.println(query.qid+ " "+"\t" + returnNQC(score, noOfTopDocs));

                }
                else
                    nqc = 0;

                for (String queryTerm : queryTerms) {
                    //System.out.println("<" + queryTerm + ">");
                    q = queryParser.parse(queryTerm);
                    //System.out.println(q.toString());

                    // get all the NNs for that particular query term to be clustered
                    List<TermSimilarity> nnTerms = new ArrayList<>();
                    nnTerms = termNNHash.get(queryTerm);
                    if(nnTerms != null) {
                    // then, get and put all NNs' vector in the list of terms to be clustered
                        for(TermSimilarity nnTerm : nnTerms) {
                            String nnTermStr = nnTerm.getTerm();
                            // +++ search for the term to get the vector:
                            q = queryParser.parse(nnTermStr);
                            //System.out.println(q.toString());
                            collector = TopScoreDocCollector.create(1);
                            vecSearcher.search(q, collector);
                            topDocs = collector.topDocs();
                            hits = topDocs.scoreDocs;
                            //System.out.println("vector search: hits.length: "+hits.length);
                            if(hits.length != 0) {
                                hashNNTerms.add(new WordVec(vecSearcher.doc(hits[0].doc).get(wordVector)));
            //                    System.out.println("Vector: "+vecSearcher.doc(hits[0].doc).get(wordVector));
                            }
                            // ---
                        }
                    }

                    /*
                    // +++ searching for cluster id of the terms
                    collector = TopScoreDocCollector.create(1);
                    clusterInfoSearcher.search(q, collector);
                    topDocs = collector.topDocs();
                    hits = topDocs.scoreDocs;
                    System.out.println("ClusterId search: hits.length: "+hits.length);
                    if(hits.length != 0) {
                        //System.out.println("ClusterId: "+clusterInfoSearcher.doc(hits[0].doc).get(wordClusterId));
                    }
                    // ---
                    // */
                } // for each query terms

                // Now, for each composition
                for(int i=0; i<queryTerms.length-1; i++) {
                    q = queryParser.parse(queryTerms[i] + " " + queryTerms[i+1]);
                    //System.out.println(q.toString());

                    // get all the NNs for that particular query term to be clustered
                    List<TermSimilarity> nnTerms = new ArrayList<>();
                    nnTerms = termNNHash.get(queryTerms[i] + " " + queryTerms[i+1]);
                    if(nnTerms != null) {
                    // then, get and put all NNs' vector in the list of terms to be clustered
                        for(TermSimilarity nnTerm : nnTerms) {
                            String nnTermStr = nnTerm.getTerm();
                            // +++ search for the term to get the vector:
                            q = queryParser.parse(nnTermStr);
                            //System.out.println(q.toString());
                            collector = TopScoreDocCollector.create(1);
                            vecSearcher.search(q, collector);
                            topDocs = collector.topDocs();
                            hits = topDocs.scoreDocs;
                            //System.out.println("vector search: hits.length: "+hits.length);
                            if(hits.length != 0) {
                                hashNNTerms.add(new WordVec(vecSearcher.doc(hits[0].doc).get(wordVector)));
            //                    System.out.println("Vector: "+vecSearcher.doc(hits[0].doc).get(wordVector));
                            }
                            // ---
                        }
                    }

                    
                }

    //            System.out.print(query.qid+ ": " +luceneQuery.toString(fieldToSearch)+"\t");

                // +++ clustering the NN terms
                listNNTerms = new ArrayList<>(hashNNTerms);
                int numWords = listNNTerms.size();
                localNumClusters = numWords / desirableClusterSize;
    //            localNumClusters = 2;
                KMeansPlusPlusClusterer<WordVec> wordClusterer = new KMeansPlusPlusClusterer<>(localNumClusters);

                // reading the word-vectors in wordList
    //            System.out.print("Clustering NNs using K-Means. K = "+localNumClusters);
                List<CentroidCluster<WordVec>> wordClusters = wordClusterer.cluster(listNNTerms);
    //            System.out.println(" Done");
                //Silhouette.returnSilhouette(wordClusters);
                Silhouette.computeSilhouette(wordClusters);
                // --- clustering the terms

                double a[] = new double[wordClusters.size()];
                int c = 0;
                double maxClusterSize = -1;
                for (CentroidCluster<WordVec> cluster : wordClusters) {
                    List<WordVec> clusterPoints = cluster.getPoints();
                    if(maxClusterSize < (double) clusterPoints.size())
                        maxClusterSize = (double) clusterPoints.size();
                    a[c++] = (double) clusterPoints.size();
    //                System.out.println(clusterPoints.size());
                }
                StandardDeviation o;
                o = new StandardDeviation();
                qpp = o.evaluate(a) / maxClusterSize;
                if(queryAP == null) {
                    //System.out.println("NULL");
                }
                else {
                    System.out.println(query.qid+"\t" + queryAP.toString() + "\t" + (alpha * qpp + (1-alpha) * nqc));
                }
                allAP[processedQueryCount] = queryAP;
                allQPP[processedQueryCount] = (alpha * qpp + (1-alpha) * nqc);
                processedQueryCount++;
            }
        }
        PearsonsCorrelation cor = new PearsonsCorrelation();
        KendallsCorrelation tau = new KendallsCorrelation();
        System.out.println("RH: "+cor.correlation(allAP, allQPP) + " TAU: "
            + tau.correlation(allAP, allQPP));
    }

    public static void main(String[] args) throws Exception {

        Approach2 obj;

        if(args.length != 0)
            obj = new Approach2(args[0]);
        else {
//            System.out.println("Usage: java Approach2 <properties.file>");
            args = new String[1];
//            args[0] = "qpp-trec1.properties";
//            args[0] = "qpp-trec2.properties";
//            args[0] = "qpp-trec3.properties";
//            args[0] = "qpp-trec6.properties";
//            args[0] = "qpp-trec7.properties";
//            args[0] = "qpp-trec8.properties";
            args[0] = "qpp-bm25-trec6.properties";
//            args[0] = "qpp-trec10.properties";
//            args[0] = "qpp-trecrb.properties";
//            args[0] = "qpp-trec123.properties";
//            args[0] = "qpp-wt10g.properties";
//            args[0] = "qpp-trec678rb.properties";
            obj = new Approach2(args[0]);
        }

        // for development set:
        int i;
        double j;
//        for (j=0.0; j<1.0; j+=0.1)
//            for (i=5; i<=15; i++)
            {
//                obj.alpha = j;
//                obj.desirableClusterSize = i;
//                obj.alpha = 1;
                System.out.println("Alpha set to "+obj.alpha+" ");
                System.out.println("desirableClusterSize set to "+obj.desirableClusterSize);
                obj.processAll();
            }
    }
}