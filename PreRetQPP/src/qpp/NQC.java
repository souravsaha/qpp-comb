/**
 * Predicting Query Performance by Query-Drift Estimation
 * Anna Shtok, Oren Kurland, and David Carmel
 * In ICTIR 2009, TOIS 2012.
 */
package qpp;

import common.InputOutput;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Properties;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import common.TRECQueryParser;
import common.TRECQuery;
import java.util.HashMap;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.xml.sax.SAXException;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;


public class NQC {

    Properties prop;
    String propPath;
    IndexReader vecReader;
    IndexReader clusterInfoReader;
    IndexSearcher vecSearcher;
    IndexSearcher clusterInfoSearcher;

    Analyzer analyzer;
    String queryPath;
    List<TRECQuery> queries;
    int numVocabClusters;
    TRECQueryParser trecQueryparser;
    String fieldToSearch;
    String wordClusterId, wordVector;
    HashMap<String, List<TermSimilarity>> termNNHash;
    private final String resPath;
    private final String mapPath;
    private final int noOfTopDocs;

    public NQC(String propPath) throws IOException, SAXException, Exception {
        
        this.propPath = propPath;
        prop = new Properties();
        try {
            prop.load(new FileReader(propPath));
        } catch (IOException ex) {
            System.err.println("Error: Properties file missing in "+propPath);
            System.exit(1);
        }
        //----- Properties file loaded

        /* setting query path */
        queryPath = prop.getProperty("queryPath");
        /* constructing the query */
        fieldToSearch = "wordname";
        trecQueryparser = new TRECQueryParser(queryPath, analyzer, fieldToSearch);
        queries = trecQueryparser.constructQueries();
        /* constructed the query */

        resPath = prop.getProperty("resPath");
        mapPath = prop.getProperty("mapPath");

        // no of top docs to be used for NQC
        noOfTopDocs = Integer.parseInt(prop.getProperty("noOfTopDocs", "100"));
    }

    public static double returnNQC(double score[], int noOfTopDocs) {
        
        double s[];

        if(score.length < noOfTopDocs)
            noOfTopDocs = score.length;
        else
            noOfTopDocs = 100;
        s = new double[noOfTopDocs];

        // copies score[0 ... noOfTopDocs-1] to s
        System.arraycopy(score, 0, s, 0, noOfTopDocs);

        StandardDeviation obj;
        obj = new StandardDeviation();
        
        return (obj.evaluate(s) / s[0]);
    }

    private void processAll() throws Exception {

        double allAP[], allQPP[];
        int processedQueryCount = 0;
        allAP = new double [queries.size()];
        allQPP = new double [queries.size()];

        HashMap<String, Double[]> queryScoreHash = InputOutput.readScoreFromResFile(this.resPath);
        HashMap<String, Double> queryMapHash = InputOutput.readAPFromFile(this.mapPath);

        for (TRECQuery query : queries) {

            Double[] qScore = queryScoreHash.get(query.qid);
            Double queryAP = queryMapHash.get(query.qid);

            if (null != qScore && null != queryAP) {
                double score[] = ArrayUtils.toPrimitive(qScore);

                double nqc = returnNQC(score, noOfTopDocs);
                System.out.println(query.qid+ " "+ 
                    "\t" + queryAP.toString()+ 
                    "\t" + nqc);
                allAP[processedQueryCount] = queryAP;
                allQPP[processedQueryCount] = nqc;
                processedQueryCount++;

            }
        }
        PearsonsCorrelation cor = new PearsonsCorrelation();
        KendallsCorrelation tau = new KendallsCorrelation();
        System.out.println("ROH, Tau: "+cor.correlation(allAP, allQPP)+ " "+ tau.correlation(allAP, allQPP));
    }

    public static void main(String[] args) throws Exception {

        NQC obj;

        if(args.length != 0)
            obj = new NQC(args[0]);
        else {
//            System.out.println("Usage: java NQC <properties.file>");
            args = new String[1];
//            args[0] = "qpp-trec2.properties";
            args[0] = "qpp-trec678rb.properties";
//            args[0] = "qpp-wt10g.properties";
            obj = new NQC(args[0]);
        }
        
        obj.processAll();
    }

}
