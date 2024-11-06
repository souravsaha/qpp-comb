/**
 * Effective Pre-retrieval Query Performance Prediction 
 * Using Similarity and Variability Evidence
 * Ying Zhao, Falk Scholer, and Yohannes Tsegay
 * IN ECIR 2008.
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
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import common.TRECQueryParser;
import common.TRECQuery;
import java.util.HashMap;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopScoreDocCollector;
import static qpp.NQC.returnNQC;



public class SCQ {

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

    public SCQ(String propPath) throws Exception {
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

        searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new CustomSimilarity());   // search of the documents containing the query terms.

        resPath = prop.getProperty("resPath");
        mapPath = prop.getProperty("mapPath");
        // no of top docs to be used for NQC
        noOfTopDocs = Integer.parseInt(prop.getProperty("noOfTopDocs", "100"));
        alpha = Double.parseDouble(prop.getProperty("alpha", "0.6"));
//        System.out.println("Alpha set to "+alpha);

    }

    private List<common.TRECQuery> constructQueries() throws Exception {

        trecQueryparser.queryFileParse();
        return trecQueryparser.queries;
    }

    public double getCollectionFrequence(String term, IndexReader reader, String fieldName) throws IOException {
            
        Term termInstance = new Term(fieldName, term);
        long termFreq = reader.totalTermFreq(termInstance); // CF: Returns the total number of occurrences of term across all documents (the sum of the freq() for each doc that has this term).

        return termFreq;
    }

    public double getDocumentFrequence(String term, IndexReader reader, String fieldName) throws IOException {
            
        Term termInstance = new Term(fieldName, term);
        double docCount = reader.docFreq(termInstance);       // DF: Returns the number of documents containing the term

        return docCount;
    }

    public long getNumDocs(IndexReader indexReader) {
        return indexReader.numDocs();
    }
    
    public double getCollectionProbability(String term, IndexReader reader, String fieldName) throws IOException {

        colDocCount = reader.maxDoc();      // total number of documents in the index
        collector = TopScoreDocCollector.create(colDocCount);

        Term termInstance = new Term(fieldName, term);
        long termFreq = reader.totalTermFreq(termInstance); // CF: Returns the total number of occurrences of term across all documents (the sum of the freq() for each doc that has this term).
        long docCount = reader.docFreq(termInstance);       // DF: Returns the number of documents containing the term

		// +++ COLLECTION SIZE
        Fields fields = MultiFields.getFields(reader);
        Terms terms = fields.terms(fieldName);
        if(null == terms) {
            System.err.println("Field: "+fieldName);
            System.err.println("Error buildCollectionStat(): terms Null found");
        }
        long colSize = terms.getSumTotalTermFreq(); // COLL-SIZE: total number of terms in the index in that field
		// --- COLLECTION SIZE

        // idf = log(#docCount / (df+1) )
        double idf = Math.log((double)(colDocCount)/(double)(docCount+1));

//        System.out.print(termFreq + " " + colSize+ " ");
//        System.out.println(idf);
        return (double)idf;
    }

    private void processAll() throws Exception {

        String queryTerms[];
        double nqc;
        double allAP[];
        double allSCQ[], allMaxSCQ[], allNSCQ[];
        double SCQ_NQC[], MaxSCQ_NQC[], NSCQ_NQC[];
        double allFinalScore[];
        double allVar[], allAvgVar[], allMaxVar[];
        double SumVar_NQC[], AvgVar_NQC[], MaxVar_NQC[];
        int processedQueryCount = 0;

        allAP = new double [queries.size()];

        allSCQ = new double [queries.size()];
        allNSCQ = new double [queries.size()];
        allMaxSCQ = new double [queries.size()];
        SCQ_NQC = new double [queries.size()];
        NSCQ_NQC = new double [queries.size()];
        MaxSCQ_NQC = new double [queries.size()];

        allVar = new double [queries.size()];
        allAvgVar = new double [queries.size()];
        allMaxVar = new double [queries.size()];
        SumVar_NQC = new double [queries.size()];
        AvgVar_NQC = new double [queries.size()];
        MaxVar_NQC = new double [queries.size()];

        allFinalScore = new double [queries.size()];

        colDocCount = reader.maxDoc();      // total number of documents in the index

        HashMap<String, Double[]> queryScoreHash = InputOutput.readScoreFromResFile(this.resPath);
        HashMap<String, Double> queryAPHash = InputOutput.readAPFromFile(this.mapPath);

        for (TRECQuery query : queries) {
            System.out.print(query.qid + "\t");
            Double[] qScore = queryScoreHash.get(query.qid);
            Double queryAP = queryAPHash.get(query.qid);

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

                double idf;
                double cf;

                double scq = 0;
                double nscq = 0;         // normalized scq
                double maxScq = 0;      // max scq

                double sumVar = 0;
                double avgVar = 0;
                double maxVar = 0;

                int qtCount = 0; // query term count

                for (String queryTerm : queryTerms) {

                    qtCount++;

                    idf = getCollectionProbability(queryTerm, reader, fieldToSearch);
                    cf = getCollectionFrequence(queryTerm, reader, fieldToSearch);

                    double df = getDocumentFrequence(queryTerm, reader, fieldToSearch);
                    double scqq = 0;
                    if(df > 0) {
                        scqq = ((1+Math.log(cf)) * Math.log((1+(colDocCount/df))));  // single query term sqc score

                        if(scqq > maxScq)
                            maxScq = scqq;

                        scq += scqq;
//                    System.out.println("\t"+scqq);

                        // +++ sigma1 calculation

                        // + wBar calculation
                        double wBar = 0;
                        Query termQuery = trecQueryparser.getAnalyzedQuery(queryTerm);
                        searcher.search(termQuery, collector);

                        ScoreDoc[] hits = null;
                        hits = collector.topDocs().scoreDocs;
                        int hits_length = hits.length;
    //                    System.out.println(hits_length);
                        for (int i = 0; i < hits_length; ++i) {
                            long termFreq = (long) hits[i].score;
                            wBar += (1+Math.log(termFreq)*Math.log(1+colDocCount / df));
                        }

                        wBar = wBar / hits_length;
    //                    System.out.println(wBar);
                        // - wBar calculation

                        double sumVarq = 0;
                        for (int i = 0; i < hits_length; ++i) {
                            long termFreq = (long) hits[i].score;
                            sumVarq += Math.pow(((1+Math.log(termFreq)*Math.log(1+colDocCount / df)) - wBar), 2);
                        }
                        sumVarq = Math.sqrt(sumVarq / df);
                        if(sumVarq > maxVar)
                            maxVar = sumVarq;
                        sumVar += sumVarq;
                        // --- sigma1 calculation
                    }
                } // ends for each query term
                nscq = scq / qtCount;
                System.out.print(scq + "\t" + maxScq + "\t" + nscq + "\t");

                avgVar = sumVar / qtCount;
                System.out.println(sumVar + "\t" + avgVar + "\t" + maxVar);

//                System.out.println(scq);
                allAP[processedQueryCount] = queryAP;

                allSCQ[processedQueryCount] = scq;
                allNSCQ[processedQueryCount] = nscq;
                allMaxSCQ[processedQueryCount] = maxScq;

//                SCQ_NQC[processedQueryCount] = (alpha * scq + (1-alpha) * nqc);
//                NSCQ_NQC[processedQueryCount] = (alpha * nscq + (1-alpha) * nqc);
//                MaxSCQ_NQC[processedQueryCount] = (alpha * maxScq + (1-alpha) * nqc);

                allVar[processedQueryCount] = sumVar;
                allAvgVar[processedQueryCount] = avgVar;
                allMaxVar[processedQueryCount] = maxVar;

//                SumVar_NQC[processedQueryCount] = (alpha * sumVar + (1-alpha) * nqc);
//                AvgVar_NQC[processedQueryCount] = (alpha * avgVar + (1-alpha) * nqc);
//                MaxVar_NQC[processedQueryCount] = (alpha * maxVar + (1-alpha) * nqc);

//                allFinalScore[processedQueryCount] = (alpha * maxVar + (1-alpha) * nqc);

                processedQueryCount++;
            }
        } // ends for each query 

        PearsonsCorrelation cor = new PearsonsCorrelation();
        KendallsCorrelation tau = new KendallsCorrelation();
        SpearmansCorrelation sper = new SpearmansCorrelation();

//        System.out.println();
//
//        System.out.println("SCQ "+cor.correlation(allAP, allSCQ) + " "
//            + tau.correlation(allAP, allSCQ) + " " + sper.correlation(allAP, allSCQ));
//        System.out.println("NSCQ "+cor.correlation(allAP, allNSCQ) + " "
//            + tau.correlation(allAP, allNSCQ) + " " + sper.correlation(allAP, allNSCQ));
//        System.out.println("MaxSCQ "+cor.correlation(allAP, allMaxSCQ) + " "
//            + tau.correlation(allAP, allMaxSCQ)  + " " + sper.correlation(allAP, allMaxSCQ) + "\n");
//
//        System.out.println("SCQ,NQC "+cor.correlation(allAP, SCQ_NQC) + " "
//            + tau.correlation(allAP, SCQ_NQC) + " " + sper.correlation(allAP, SCQ_NQC));
//        System.out.println("NSCQ,NQC "+cor.correlation(allAP, NSCQ_NQC) + " "
//            + tau.correlation(allAP, NSCQ_NQC) + " " + sper.correlation(allAP, NSCQ_NQC));
//        System.out.println("MaxSCQ,NQC "+cor.correlation(allAP, MaxSCQ_NQC) + " "
//            + tau.correlation(allAP, MaxSCQ_NQC) + " " + sper.correlation(allAP, MaxSCQ_NQC) + "\n");
//
//        System.out.println("SumVAR "+cor.correlation(allAP, allVar) + " "
//            + tau.correlation(allAP, allVar) + " " + sper.correlation(allAP, allVar));
//        System.out.println("AvgVAR "+cor.correlation(allAP, allAvgVar) + " "
//            + tau.correlation(allAP, allAvgVar) + " " + sper.correlation(allAP, allAvgVar));
//        System.out.println("MaxVAR "+cor.correlation(allAP, allMaxVar) + " "
//            + tau.correlation(allAP, allMaxVar) + " " + sper.correlation(allAP, allMaxVar) + "\n");
//
//        System.out.println("VAR,NQC "+cor.correlation(allAP, SumVar_NQC) + " "
//            + tau.correlation(allAP, SumVar_NQC) + " " + sper.correlation(allAP, SumVar_NQC));
//        System.out.println("AvgVAR,NQC "+cor.correlation(allAP, AvgVar_NQC) + " "
//            + tau.correlation(allAP, AvgVar_NQC) + " " + sper.correlation(allAP, AvgVar_NQC));
//        System.out.println("MaxVAR,NQC "+cor.correlation(allAP, MaxVar_NQC) + " "
//            + tau.correlation(allAP, MaxVar_NQC) + " " + sper.correlation(allAP, MaxVar_NQC) + "\n");
    }

    public static void main(String[] args) throws Exception {
        
        SCQ obj;
        if(args.length != 0)
            obj = new SCQ(args[0]);
        else {
//            System.out.println("Usage: java Approach2 <properties.file>");
            args = new String[1];
            args[0] = "clueweb.properties";
            obj = new SCQ(args[0]);
            obj.alpha = 0.1;
//            for (obj.alpha = 0.0; obj.alpha<=1.0; obj.alpha+=0.1) 
            {
                System.out.println("Alpha set to " + obj.alpha + " ");
                obj.processAll();
                System.out.println();
            }
        }

    }

}
