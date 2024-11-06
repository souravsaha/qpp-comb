/**
 * Using Coherence-Based Measures to Predict Query Difficulty
 * Jiyin He, Martha Larson, and Maarten de Rijke
 * ECIR - 2008
 * 
 * Methods: 
 *   1 - Average Query Term Coherence - AvQC
 *   2 - Average Query Term Coherence with Global Constraint - AvQCG
 * .
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
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopScoreDocCollector;



public class QC {

    String          propPath;
    Properties      prop;
    IndexReader     reader;
    IndexSearcher   searcher;
    String          indexPath;
    File            indexFile;
    Analyzer        analyzer;
    String          stopFilePath;
    String          queryPath;
    File            queryFile;      // the query file
    int             queryFieldFlag; // 1. title; 2. +desc, 3. +narr
    String fieldToSearch;
    List<common.TRECQuery> queries;
    TRECQueryParser trecQueryparser;
    int colDocCount;    // collection document count
    String resPath;
    private String mapPath;
    int noOfTopDocs;
    double alpha;
    TopScoreDocCollector collector;
    
    CosineSimilarity cosineSim;

    // the setCoherence
    double simThreshold;     // if(cos-sim(di,dj) > simThreshold) \delta(di, dj) = 1; else \delta(di, dj) = 0;

    public QC(String propPath) throws Exception {
        this.propPath = propPath;
        prop = new Properties();
        try {
            prop.load(new FileReader(propPath));
        } catch (IOException ex) {
            System.err.println("Error: Properties file missing in "+propPath);
            System.exit(1);
        }
        //----- Properties file loaded

        // +++++ setting the analyzer with English Analyzer with Smart stopword list
        stopFilePath = prop.getProperty("stopFilePath");
        common.EnglishAnalyzerWithSmartStopword engAnalyzer = new common.EnglishAnalyzerWithSmartStopword(stopFilePath);
        analyzer = engAnalyzer.setAndGetEnglishAnalyzerWithSmartStopword();
        // ----- analyzer set: analyzer

        //+++++ index path setting 
        indexPath = prop.getProperty("indexPath");
        indexFile = new File(indexPath);
        Directory indexDir = FSDirectory.open(indexFile.toPath());

        if (!DirectoryReader.indexExists(indexDir)) {
            System.err.println("Index doesn't exists in "+indexPath);
            System.exit(1);
        }
        //----- index path set

        /* setting query path */
        queryPath = prop.getProperty("queryPath");
        queryFile = new File(queryPath);
        /* query path set */
        fieldToSearch = prop.getProperty("fieldToSearch");//"full-content";

        /* constructing the query */
        trecQueryparser = new common.TRECQueryParser(queryPath, analyzer, fieldToSearch);
        queries = constructQueries();
        /* constructed the query */

        /* setting reader and searcher */
        reader = DirectoryReader.open(FSDirectory.open(indexFile.toPath()));

        colDocCount = reader.maxDoc();      // total number of documents in the index
        collector = TopScoreDocCollector.create(colDocCount);

        searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new CustomSimilarity());   // search of the documents containing the query terms.

        resPath = prop.getProperty("resPath");
        mapPath = prop.getProperty("mapPath");
        // no of top docs to be used for NQC
        noOfTopDocs = Integer.parseInt(prop.getProperty("noOfTopDocs", "100"));
        alpha = Double.parseDouble(prop.getProperty("alpha", "0.6"));
//        System.out.println("Alpha set to "+alpha);

        cosineSim = new CosineSimilarity(reader, fieldToSearch);
        simThreshold = 0.2;
    }

    private List<common.TRECQuery> constructQueries() throws Exception {

        trecQueryparser.queryFileParse();
        return trecQueryparser.queries;
    }

    private void processAll() throws Exception {

        String queryTerms[];
        double nqc;
        double allAP[];
        double avQC_NQC[];
        double avQCG_NQC[];
        double allFinalScore[];
        int processedQueryCount = 0;

        allAP = new double [queries.size()];

        avQC_NQC = new double [queries.size()];
        avQCG_NQC = new double [queries.size()];
        allFinalScore = new double [queries.size()];

        colDocCount = reader.maxDoc();      // total number of documents in the index

        HashMap<String, Double[]> queryScoreHash = InputOutput.readScoreFromResFile(this.resPath);
        HashMap<String, Double> queryAPHash = InputOutput.readAPFromFile(this.mapPath);

        for (TRECQuery query : queries) {
            System.out.print(query.qid + "\t");
            Double[] qScore = queryScoreHash.get(query.qid);
            Double queryAP = queryAPHash.get(query.qid);
            Set<Integer> allRetrievedDocs = new HashSet<>();    // to be used for global SetCoherence calculation

            Query luceneQuery = trecQueryparser.getAnalyzedQuery(query);
            if(!luceneQuery.toString(fieldToSearch).isEmpty()) {
//                System.out.println(luceneQuery.toString(fieldToSearch));
                queryTerms = new String[luceneQuery.toString(fieldToSearch).split(" ").length];
                queryTerms = luceneQuery.toString(fieldToSearch).split(" ");

//                if (null != qScore && null != queryAP) {
//                    double score[] = ArrayUtils.toPrimitive(qScore);
//
//                    nqc = returnNQC(score, noOfTopDocs);
//    //                System.out.println(query.qid+ " "+"\t" + returnNQC(score, noOfTopDocs));
//                }
//                else
//                    nqc = 0;

                int qtCount = 0; // query term count

                double setCoherence = 0;
                double avQC = 0;
                double avQCG = 0;
                int coherenceCount;
                int noDocsConsidered;
                int hits_list[];

                for (String queryTerm : queryTerms) {

//                    System.out.println(queryTerm);
                    qtCount++;

                    Query termQuery = trecQueryparser.getAnalyzedQuery(queryTerm);
                    collector = TopScoreDocCollector.create(colDocCount);

                    searcher.search(termQuery, collector);

                    ScoreDoc[] hits = null;
                    hits = collector.topDocs().scoreDocs;
                    int hits_length = hits.length;
//                    System.out.println("hits_length " + hits_length);

                    noDocsConsidered = Math.min(50,hits_length);
                    if (hits_length > 2 ) {
                    // for each of document, containing the query term
                        hits_list = new int[noDocsConsidered];
                        for (int i = 0; i < noDocsConsidered; ++i)
                            hits_list[i] = hits[i].doc;

                        for (int i = 0; i < noDocsConsidered; ++i)
                            allRetrievedDocs.add(hits_list[i]);

                        coherenceCount = 0;
                        for (int i = 0; i < noDocsConsidered; ++i) {
                            for (int j = 0; j< noDocsConsidered; ++j) {
                                if (i != j) {
                                    if(cosineSim.computeCosineSimilarity(hits_list[i], hits_list[j]) >= simThreshold)
                                        coherenceCount ++;
                                }
                            }
                        }
                        setCoherence += (coherenceCount / 
                            (double)(noDocsConsidered * (noDocsConsidered-1)));
                    }
//                    System.out.println(setCoherence);
                } // ends for each query term
                if(qtCount <=0)
                    System.in.read();
                avQC = setCoherence / qtCount;
                System.out.print(avQC + "\t");

                // + calculating setCoherence for the entire query Q
                setCoherence = 0;
                coherenceCount = 0;
                int allRetDocsCount = allRetrievedDocs.size();
                hits_list = new int[allRetDocsCount];
                int k = 0;
                for(Integer in : allRetrievedDocs)
                    hits_list[k++] = in;
                noDocsConsidered = Math.min(100,allRetDocsCount);
                for (int i = 0; i <noDocsConsidered ; ++i) {
                    for (int j = 0; j< allRetDocsCount; ++j) {
                        if (i != j) {
                            if(cosineSim.computeCosineSimilarity(hits_list[i], hits_list[j]) >= simThreshold)
                                coherenceCount ++;
                        }
                    }
                }
                setCoherence += (coherenceCount / 
                    (double)(noDocsConsidered * (noDocsConsidered-1)));
                // - calculating setCoherence for the entire query Q
                avQCG = avQC * simThreshold;
                System.out.println(avQCG);
                allAP[processedQueryCount] = queryAP;

//                allFinalScore[processedQueryCount] = (alpha * maxVar + (1-alpha) * nqc);
//                avQC_NQC[processedQueryCount] = (alpha * avQC  + (1-alpha) * nqc);
//                avQCG_NQC[processedQueryCount] = (alpha * avQCG  + (1-alpha) * nqc);

                processedQueryCount++;
            }
        } // ends for each query 

        PearsonsCorrelation cor = new PearsonsCorrelation();
        KendallsCorrelation tau = new KendallsCorrelation();

//        System.out.println();
//        System.out.println("AvQC,NQC "+cor.correlation(allAP, avQC_NQC) + " "
//            + tau.correlation(allAP, avQC_NQC));
//        System.out.println("AvQCG,NQC "+cor.correlation(allAP, avQCG_NQC) + " "
//            + tau.correlation(allAP, avQCG_NQC));
//        System.out.println();

    }

    public static void main(String[] args) throws Exception {
        
        QC obj;
        if(args.length != 0)
            obj = new QC(args[0]);
        else {
//            System.out.println("Usage: java Approach2 <properties.file>");
            args = new String[1];
            args[0] = "clueweb.properties";
            obj = new QC(args[0]);
//            obj.alpha = 1.0;
//            System.out.println("Alpha set to " + obj.alpha + " ");
//            for(float i=0.1f; i<0.9; i+=0.1)
//            {
//                obj.simThreshold = i;
//                System.out.println("SimThreshold = " + obj.simThreshold);
//                obj.processAll();
//                System.out.println();
//            }
//            TREC3
            // AvQC*
//            obj.alpha = 0.1; obj.simThreshold = 0.3;
            // AvQCG*
//            obj.alpha = 0.3; obj.simThreshold = 0.3;
//            for (obj.alpha = 0.0; obj.alpha<=1.0; obj.alpha+=0.1) 
            // AvQC, AvQCG
            obj.alpha = 1.0; obj.simThreshold = 0.3;
//            {
//                for(obj.simThreshold=0.1f; obj.simThreshold<=0.9; obj.simThreshold+=0.1)
//                {
//                    System.out.println("Alpha: " + obj.alpha + " SimThreshold: " + obj.simThreshold);
                    obj.processAll();
//                    System.out.println();
//                }
//            }
        }

    }

}
