/**
 * AvgIDF:
 * Predicting query performance
 * Steve Cronen-Townsend, Yun Zhou, and Bruce Croft
 * In SIGIR 2002
 * ============
 * MaxIDF:
 * Query association surrogates for web search
 * Scholer, Williams, and Turpin
 * JASIST, 55(7):637-650, 2004
 * ============.
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
import org.apache.lucene.search.Query;
import static qpp.NQC.returnNQC;


public class MaxIDF {

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

    public MaxIDF(String propPath) throws Exception {
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

        resPath = prop.getProperty("resPath");
        mapPath = prop.getProperty("mapPath");
        // no of top docs to be used for NQC
        noOfTopDocs = Integer.parseInt(prop.getProperty("noOfTopDocs", "100"));
        alpha = Double.parseDouble(prop.getProperty("alpha", "1.0"));
//        System.out.println("Alpha set to "+alpha);

    }

    private List<common.TRECQuery> constructQueries() throws Exception {

        trecQueryparser.queryFileParse();
        return trecQueryparser.queries;
    }

    public float getCollectionProbability(String term, IndexReader reader, String fieldName) throws IOException {

        colDocCount = reader.maxDoc();      // total number of documents in the index

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
        double idf = Math.log((float)(colDocCount)/(float)(docCount+1));

//        System.out.print(termFreq + " " + colSize+ " ");
//        System.out.println(idf);
        return (float)idf;
    }

    private void processAll() throws Exception {

        int processedQueryCount = 0;
        String queryTerms[];
        float maxQIdf = -1;
        float maxIdf = 0;
        float avgIdf = 0;
        double nqc;
        double allAP[], allQPP[];
        double AvgIDF_NQC[];
        double MaxIDF_NQC[];

        allAP = new double [queries.size()];
        allQPP = new double [queries.size()];

        AvgIDF_NQC = new double [queries.size()];
        MaxIDF_NQC = new double [queries.size()];

        HashMap<String, Double[]> queryScoreHash = InputOutput.readScoreFromResFile(this.resPath);
        HashMap<String, Double> queryAPHash = InputOutput.readAPFromFile(this.mapPath);

        for (TRECQuery query : queries) {
            System.out.print(query.qid + "\t");
            Query luceneQuery = trecQueryparser.getAnalyzedQuery(query);
            if(!luceneQuery.toString(fieldToSearch).isEmpty()) {
                queryTerms = new String[luceneQuery.toString(fieldToSearch).split(" ").length];
                queryTerms = luceneQuery.toString(fieldToSearch).split(" ");

                maxIdf = 0;
                avgIdf = 0;
                float idf;
                for (String queryTerm : queryTerms) {
                    idf = getCollectionProbability(queryTerm, reader, fieldToSearch);
                    avgIdf += idf;
                    if(idf > maxIdf)
                        maxIdf = idf;
                }
//                System.out.println(maxIdf/ (float)colDocCount + " " + avgIdf / queryTerms.length);
                System.out.println(maxIdf + "\t" + avgIdf / queryTerms.length);
            }
            maxQIdf = (maxQIdf < maxIdf)?maxIdf:maxQIdf;
        }

//        System.out.println();
        for (TRECQuery query : queries) {
//            System.out.print(query.qid + " ");
            Double[] qScore = queryScoreHash.get(query.qid);
            Double queryAP = queryAPHash.get(query.qid);

            Query luceneQuery = trecQueryparser.getAnalyzedQuery(query);
            if(!luceneQuery.toString(fieldToSearch).isEmpty()) {
                queryTerms = new String[luceneQuery.toString(fieldToSearch).split(" ").length];
                queryTerms = luceneQuery.toString(fieldToSearch).split(" ");

                if (null != qScore && null != queryAP) {
                    double score[] = ArrayUtils.toPrimitive(qScore);

                    nqc = returnNQC(score, noOfTopDocs);
    //                System.out.println(query.qid+ " "+"\t" + returnNQC(score, noOfTopDocs));

                }
                else
                    nqc = 0;

                maxIdf = 0;
                avgIdf = 0;
                float idf;
                for (String queryTerm : queryTerms) {
                    idf = getCollectionProbability(queryTerm, reader, fieldToSearch);
                    avgIdf += idf;
                    if(idf > maxIdf)
                        maxIdf = idf;
                }
//                System.out.println(maxIdf/ maxQIdf + " " + avgIdf / queryTerms.length);
//                System.out.println( + " " + avgIdf / queryTerms.length);
//                System.out.println(query.qid+"\t" + queryAP.toString() + "\t" + (alpha * nqc + (1-alpha) * maxIdf/ maxQIdf));
                allAP[processedQueryCount] = queryAP;
                allQPP[processedQueryCount] = (alpha * maxIdf / queryTerms.length + (1-alpha) * nqc);
                MaxIDF_NQC[processedQueryCount] = (alpha * maxIdf / queryTerms.length + (1-alpha) * nqc);
                AvgIDF_NQC[processedQueryCount] = (alpha * avgIdf / queryTerms.length + (1-alpha) * nqc);
                processedQueryCount++;

            }
        }
        PearsonsCorrelation cor = new PearsonsCorrelation();
        KendallsCorrelation tau = new KendallsCorrelation();
//        System.out.println(" "+cor.correlation(allAP, allQPP) + " "
//            + tau.correlation(allAP, allQPP));

//        System.out.println("AvgIDF,NQC "+cor.correlation(allAP, AvgIDF_NQC) + " "
//            + tau.correlation(allAP, AvgIDF_NQC));
//        System.out.println("MaxIDF,NQC "+cor.correlation(allAP, MaxIDF_NQC) + " "
//            + tau.correlation(allAP, MaxIDF_NQC));
    }

    public static void main(String[] args) throws Exception {
        
        MaxIDF obj;
        if(args.length != 0)
            obj = new MaxIDF(args[0]);
        else {
//            System.out.println("Usage: java Approach2 <properties.file>");
            args = new String[1];
            args[0] = "clueweb.properties";
            obj = new MaxIDF(args[0]);
            obj.alpha = 1.0;
//            obj.alpha = 0.1;
//            for (obj.alpha = 0.0; obj.alpha<=1.0; obj.alpha+=0.1) 
            {
                System.out.println("Alpha set to " + obj.alpha + " ");
                obj.processAll();
            }
        }

    }

}
