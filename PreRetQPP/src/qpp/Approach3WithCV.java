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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import org.apache.commons.lang.ArrayUtils;
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



public class Approach3WithCV {

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
    double allAP[], allQPP[];
    double rho, tau;
    int processedQueryCount;

    public Approach3WithCV(String propPath) throws IOException, SAXException, Exception {
        
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
        numVocabClusters = Integer.parseInt(prop.getProperty("wvecs.vocabCluster.numClusters", "100"));
        if (numVocabClusters > 0) {
            String clusterInfoIndexPath = prop.getProperty("wvecs.vocabCluster.baseIndexPath") + "/" + numVocabClusters;
            clusterInfoReader = DirectoryReader.open(FSDirectory.open(new File(clusterInfoIndexPath).toPath()));
            clusterInfoSearcher = new IndexSearcher(clusterInfoReader);
            clusterInfoSearcher.setSimilarity(new DefaultSimilarity());
        }
        System.out.println("NUmber of global clusters: " + numVocabClusters);
        // --- cluster index

        // +++
        termNNHash = new HashMap<>();
        String nnPath = prop.getProperty("wvecs.nnPath");
        readNNFile(nnPath);
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
        System.out.println("Lambda set to "+alpha);

        // no of top docs to be used for NQC
        noOfTopDocs = Integer.parseInt(prop.getProperty("noOfTopDocs", "100"));
    }

    /**
     * Reads NN file to put NNs of each term into an hashmap;
     * Sets the hashmap - termNNHash.
     * @param nnPath
     * @throws FileNotFoundException
     * @throws IOException 
     */
    private void readNNFile(String nnPath) throws FileNotFoundException, IOException {

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
            rootTerm_NNWithSim = line.split("\t");  // noOfNNs+1 fields; first: root-term; then noOfNNs no. of terms with similarity
            nnList = new ArrayList<>();
            String rootWord = rootTerm_NNWithSim[0];
            //System.out.println("Root word: " + rootWord + " " + rootTerm_NNWithSim.length);
            for (int i = 0; i < noOfNNs; i++) {
                String[] singleNNWithSim = rootTerm_NNWithSim[i+1].split(" ");
                nnList.add(new TermSimilarity(singleNNWithSim[0], Float.parseFloat(singleNNWithSim[1])));
            }
            termNNHash.put(rootWord, nnList);
	}
 
	br.close();

        System.out.println("NNs read into memory");
    }

    private void processOneQuery(TRECQuery query) throws Exception {

        String queryTerms[];
        TopScoreDocCollector collector;
        ScoreDoc[] hits = null;
        TopDocs topDocs;
        Term t;
        Query q;
        QueryParser queryParser = new QueryParser(fieldToSearch, new WhitespaceAnalyzer());

        List<WordVec> listNNTerms;      // list of NN terms for a particular query; to be used for clustering
        HashSet<WordVec> hashNNTerms;   // hashmap of NN terms for a particular query; to be used for making the list for clustering

        double nqc, qpp;

//        allAP = new double [queries.size()];
//        allQPP = new double [queries.size()];

        // ++ informations needed to compute NQC
        HashMap<String, Double[]> queryScoreHash = InputOutput.readScoreFromResFile(this.resPath);
        HashMap<String, Double> queryAPHash = InputOutput.readAPFromFile(this.mapPath);
        // -- informations needed to compute NQC for each query.

//        for (TRECQuery query : queries)
        {
        // for each query, 
        // take the k-NN of each of the query term and, look
        // for the corresponding global cluster-id. On the basis of the global 
        // cluster-id, see the distribution of terms in the global clusters.
        // Finally, compute the standard deviation of the distribution of the terms 
        // in the different clusters (depending on the number of terms in each cluster).

            listNNTerms = new ArrayList<>();
            hashNNTerms = new HashSet<>();

            Query luceneQuery = trecQueryparser.getAnalyzedQuery(query);
//            System.out.println(query.qid+ " " + luceneQuery.toString(fieldToSearch));

            queryTerms = new String[luceneQuery.toString(fieldToSearch).split(" ").length];
            queryTerms = luceneQuery.toString(fieldToSearch).split(" ");

            // reading information to compute NQC for this query
            Double[] qScore = queryScoreHash.get(query.qid);
            Double queryAP = queryAPHash.get(query.qid);

            if (null != qScore && null != queryAP) {
                double score[] = ArrayUtils.toPrimitive(qScore);

                nqc = returnNQC(score, noOfTopDocs);
//                System.out.println(query.qid+ " "+"\t" + returnNQC(score, noOfTopDocs));

            }
            else
                nqc = 0;
            // -- NQC computed
            
            double phi = 0;

            // HashMap keyed by global-cid, value is a list of terms in 
            // the NNs belonging to the same global-cid.
            HashMap<String, List<WordVec>> nnGlobalClusterDist = new HashMap<>();


            for (String queryTerm : queryTerms) {
                //System.out.println("<" + queryTerm + ">");
                q = queryParser.parse(queryTerm);
//                System.out.println(q.toString());

                // get all the NNs for that particular query term
                // and see their global cluster distribution

                List<TermSimilarity> nnTerms = new ArrayList<>();
                nnTerms = termNNHash.get(queryTerm);
                if(nnTerms != null) {
                // if NNs are calculated for this query term,
                // then, look for the global-cid of each nn terms and, 
                // put the term in the corresponding list
                    for(TermSimilarity nnTerm : nnTerms) {
                        String nnTermStr = nnTerm.getTerm();

                        // +++ search for the vector of the term
                        q = queryParser.parse(nnTermStr);
                        //System.out.println(q.toString());
                        collector = TopScoreDocCollector.create(1);
                        vecSearcher.search(q, collector);
                        topDocs = collector.topDocs();
                        hits = topDocs.scoreDocs;
                        //System.out.println("vector search: hits.length: "+hits.length);
                        if(hits.length != 0) {
                        // a vector is found for that term
                            hashNNTerms.add(new WordVec(vecSearcher.doc(hits[0].doc).get(wordVector)));
        //                    System.out.println("Vector: "+vecSearcher.doc(hits[0].doc).get(wordVector));
                        }
        //                /*

                        // +++ searching for global cluster id of the terms
                        collector = TopScoreDocCollector.create(1);
                        clusterInfoSearcher.search(q, collector);
                        topDocs = collector.topDocs();
                        hits = topDocs.scoreDocs;
                        //System.out.println("ClusterId search: hits.length: "+hits.length);
                        if(hits.length != 0) {
                        // a global cluster-id is found for that term
                            String clusterId = clusterInfoSearcher.doc(hits[0].doc).get(wordClusterId);
                            //System.out.println("ClusterId: " + clusterId);
                            if (null != nnGlobalClusterDist.get(clusterId)) {
                            // if there is already some terms belonging to that  
                            // global cluster-id, then put this term in the 
                            // already-made list of terms for that global cluster-id
                                nnGlobalClusterDist.get(clusterId).add(new WordVec(vecSearcher.doc(hits[0].doc).get(wordVector)));
                            }
                            else {
                            // i.e. this is the first term belonging to this cluster-id.
                            // Make a list, put the term in the list and put the
                            // list in the hashmap for that cluster-id.
                                List<WordVec> temp = new ArrayList<>();
                                temp.add(new WordVec(vecSearcher.doc(hits[0].doc).get(wordVector)));
                                nnGlobalClusterDist.put(clusterId, temp);
                            }
                        }
                    } // for each NN term
                }
            } // for each query term
            // Now compute the standard deviation of the term distribution 
            // of the NNs, according to the global cluster-ids.
//            System.out.print(query.qid+ ": " +luceneQuery.toString(fieldToSearch)+"\t");

//            System.out.println(nnGlobalClusterDist.size());
            double a[] = new double[nnGlobalClusterDist.size()];
            int c = 0;
            double maxClusterSize = -1;
            for (Map.Entry<String, List<WordVec>> entrySet : nnGlobalClusterDist.entrySet()) {
                List<WordVec> sameClusterPoints = entrySet.getValue();
                if(maxClusterSize < (double) sameClusterPoints.size())
                    maxClusterSize = (double) sameClusterPoints.size();
                a[c++] = (double) sameClusterPoints.size();
//                System.out.println(clusterPoints.size());
            }
            
            StandardDeviation o;
            o = new StandardDeviation();
            qpp = o.evaluate(a) / maxClusterSize;
            if(queryAP == null) {
                //System.out.println("NULL");
            }
            else {
//                System.out.println(query.qid+"\t" + queryAP.toString() + "\t" + (alpha * nqc + (1-alpha) * qpp));
            }
            allAP[processedQueryCount] = queryAP;
            allQPP[processedQueryCount] = (alpha * qpp + (1-alpha) * nqc);
            processedQueryCount++;
        } // ends for each query

    }

    private void processAll(List<TRECQuery> queries) throws Exception {

        allAP = new double [queries.size()];
        allQPP = new double [queries.size()];

        processedQueryCount = 0;
        for (TRECQuery query : queries) {
            processOneQuery(query);
        } // ends for each query

        PearsonsCorrelation pCor = new PearsonsCorrelation();
        KendallsCorrelation kCor = new KendallsCorrelation();
        rho = pCor.correlation(allAP, allQPP);
        tau = kCor.correlation(allAP, allQPP);
//        System.out.println("RHO: " + rho + " TAU: " + tau);
    }

    /**
     * K-Fold cross validation
     * @param K The number of rounds of cross validation.
     * @param queries List of all queries.
     */
    public void crossValidation(int K, List<TRECQuery> queries) throws Exception {

        List<TRECQuery> train;
        List<TRECQuery> test;
        double maxRhoAlpha = 0, maxTauAlpha = 0;

        int N = queries.size();

//        Collections.shuffle(queries);

        int chunk = N / K;
        double maxRho = -2, maxTau = -2;

        for (int iteration = 0; iteration < K; iteration++) {

            System.out.println("Iteration: " + iteration);
            maxRho = maxTau = -2;

            train = new ArrayList<>();
            test = new ArrayList<>();

            int start = chunk * iteration;
            int end = chunk * (iteration + 1);
            if (iteration == K-1) 
                end = N;

            for (int sample = 0; sample < N; sample++) {
                if (sample >= start && sample < end)
                    test.add(queries.get(sample));
                else
                    train.add(queries.get(sample));
            }
            // train and test sets are ready

            for (double i = 0.1; i <= 1.0; i+=0.1) 
            {
                alpha = i;
                System.out.println("Alpha: " + alpha);
                processedQueryCount = 0;

                processAll(train);

                if(maxRho < rho) {
                    maxRho = rho;
                    maxRhoAlpha = alpha;
                }
                if(maxTau < tau) {
                    maxTau = tau;
                    maxTauAlpha = alpha;
                }

            } // ends for each alpha
//            System.out.println("maxRho: " + maxRho + " maxTau + "+ maxTau);
//            System.out.println("maxRhoAlpha: " + maxRhoAlpha + " maxTauAlpha: " + maxTauAlpha);

            System.out.println("Testing:");
            alpha = maxRhoAlpha;
            processAll(test);
            System.out.println("Rho: " + rho + " Tau: "+ tau);
            
        } // ends for all K iterations

    }

    /**
     * K-Fold cross validation
     * @param K The number of rounds of cross validation.
     * @param queries List of all queries.
     */
    public void TwoFoldCV(int K, List<TRECQuery> queries) throws Exception {

        List<TRECQuery> train;
        List<TRECQuery> test;
        double maxRhoAlpha = 0, maxTauAlpha = 0;
        double avgRho, avgTau;

        int N = queries.size();

        double maxRho = -2, maxTau = -2;

        avgRho = avgTau = 0;

        for (int iteration = 0; iteration < K; iteration++) {

            System.out.println("Iteration: " + iteration);

            Collections.shuffle(queries);

            train = new ArrayList<>();
            test = new ArrayList<>();

            for (int sample = 0; sample < N/2; sample++)
                test.add(queries.get(sample));
            for (int sample = N/2; sample < N; sample++)
                train.add(queries.get(sample));
            // train and test sets are ready

            maxRho = maxTau = -2;
            for (double i = 0.1; i <= 1.0; i+=0.1) {
                alpha = i;
//                System.out.println("Alpha: " + alpha);
                processedQueryCount = 0;

                processAll(train);

                if(maxRho < rho) {
                    maxRho = rho;
                    maxRhoAlpha = alpha;
                }
                if(maxTau < tau) {
                    maxTau = tau;
                    maxTauAlpha = alpha;
                }

            } // ends for each alpha

            System.out.println("Testing:");
            alpha = maxRhoAlpha;
//            System.out.println("Alpha: " + alpha);
            processAll(test);
            System.out.println("Rho: " + rho + " Tau: "+ tau);
            avgRho += rho;
            avgTau += tau;
            
            maxRho = maxTau = -2;
            for (double i = 0.1; i <= 1.0; i+=0.1) {
                alpha = i;
//                System.out.println("Alpha: " + alpha);
                processedQueryCount = 0;

                processAll(test);

                if(maxRho < rho) {
                    maxRho = rho;
                    maxRhoAlpha = alpha;
                }
                if(maxTau < tau) {
                    maxTau = tau;
                    maxTauAlpha = alpha;
                }

            } // ends for each alpha

            System.out.println("Testing:");
            alpha = maxRhoAlpha;
//            System.out.println("Alpha: " + alpha);
            processAll(train);
            System.out.println("Rho: " + rho + " Tau: "+ tau);
            avgRho += rho;
            avgTau += tau;
            
        } // ends for all K iterations
        avgRho /= (2*K);
        avgTau /= (2*K);

        System.out.println("avgRho: "+avgRho+" avgTau: "+avgTau);
    }

    public static void main(String[] args) throws Exception {

        Approach3WithCV obj;

        if(args.length != 0)
            obj = new Approach3WithCV(args[0]);
        else {
//            System.out.println("Usage: java Approach2 <properties.file>");
            args = new String[1];
            args[0] = "qpp-trec10.properties";
            obj = new Approach3WithCV(args[0]);
        }

//        for (double i = 0.0; i <= 1.0; i+=0.1) 
//        {
//        obj.crossValidation(5, obj.queries);
        obj.TwoFoldCV(40, obj.queries);
//        }   
    }
}
