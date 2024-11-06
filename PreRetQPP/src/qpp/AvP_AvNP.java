/**
 * Linguistic features to predict query difficulty - a case study on previous TREC campaigns 
 * Josiane Mothe and Ludovic Tanguy
 * In SIGIR 2005 Workshop: Predicting Query Difficulty - Methods and Applications, 2005
 * 
 * Methods:
 *  1 - Averaged Polysemy - AvP
 *  2 - Averaged Noun Polysemy - AvNP
 * ============.
 */
package qpp;

import common.InputOutput;
import common.TRECQuery;
import common.TRECQueryParser;
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.morph.WordnetStemmer;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.lucene.analysis.Analyzer;
import org.xml.sax.SAXException;



public class AvP_AvNP {
    
    IDictionary dict;
    URL url;

    String propPath;
    Properties prop;
    Analyzer analyzer;
    String queryPath;
    List<TRECQuery> queries;
    int numVocabClusters;
    TRECQueryParser trecQueryparser;
    String fieldToSearch;
    String wordClusterId, wordVector;
    String resPath;
    String mapPath;
    int noOfTopDocs;
    double alpha;       // linear smoothing parameter, to combine with NQC
    WordnetStemmer stemmer;
    HashMap<String, Integer> stopwordList;
    WordNet wn;

    AvP_AvNP(String propPath) throws MalformedURLException, IOException, SAXException, Exception {

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

        String wnhome = "/home/suchana/NetBeansProjects/PreRetQPP/WordNet-3.0";
//        String wnhome = System.getenv("WNHOME");
        String path = wnhome + File.separator + "dict";
        url = new URL ("file", null, path) ;

        // construct the dictionary object and open it
        dict = new Dictionary ( url ) ;
        dict.open();
        stemmer = new WordnetStemmer(dict);

        // +++ making hashmap with smart-stopword list
        stopwordList = new HashMap<>();
        File file = new File("/home/suchana/smart-stopwords");
        FileReader fileReader = new FileReader(file);
        BufferedReader bufferedReader = new BufferedReader(fileReader);
        String line;
        while ((line = bufferedReader.readLine()) != null) {
            stopwordList.put(line, 1);
        }
        fileReader.close();
        // ---
        wn = new WordNet();
    }

    private void processAll() throws Exception {

        double allAP[], allAvP[], allAvNP[];
        int processedQueryCount = 0;
        allAP = new double [queries.size()];
        allAvP = new double [queries.size()];
        allAvNP = new double [queries.size()];

        HashMap<String, Double[]> queryScoreHash = InputOutput.readScoreFromResFile(this.resPath);
        HashMap<String, Double> queryAPHash = InputOutput.readAPFromFile(this.mapPath);

        int allSenseCount;
        int nounSenseCount;

        HashMap<String, Float> allAvPScores = new HashMap<>();    // to contain all the scores of all the queries for ALL senses
        HashMap<String, Float> allAvNPScores = new HashMap<>();    // to contain all the scores of all the queries for NOUN senses

        for (TRECQuery query : queries) {

            float ambScore = 0;
            allSenseCount = 0;
            nounSenseCount = 0;

//            System.out.print(query.qid + "\t");
            String[] terms = query.qtitle.replaceAll("[^a-zA-Z0-9]", " ").split(" ");
            for (String term : terms) {
//                System.out.println(term);

                if(!term.isEmpty()){
                    //System.out.print(term + " : ");
//                    List<String> stems = stemmer.findStems(term, null);
//                    if(!stems.isEmpty())
//                    {
//                        term = stems.get(0);
                        int senseCount = wn.returnAllSensesCount(term);
                        if( 0 != senseCount ) {
                            //System.out.println(term);
                            allSenseCount += senseCount;
                        }
                        else {
                            // term not found in WordNet, assigned a single sense
                            allSenseCount += 1;
                        }

                        senseCount = wn.returnNounSenses(term);
                        if( 0 != senseCount ) {
                            nounSenseCount += senseCount;
                        }
                        else {
                            // term not found in WordNet, assigned a single sense
                            nounSenseCount += 1;
                        }
//                    }
//                    else {
//                        //System.out.println("## - OOV");
//                        //System.out.println("OOV2: " + term);
//                        OOVQuerytermCount ++;
//                    }
                }
            }

            ambScore = (float)allSenseCount / (float)terms.length;

            //System.out.println(sensesCount + " " + dictQuerytermCount + " " + OOVQuerytermCount + " " + ambScore);
            allAvPScores.put(query.qid, ambScore);

            ambScore = (float)nounSenseCount / (float)terms.length;
            allAvNPScores.put(query.qid, ambScore);
        }

        double nqc;

        // AvP
        processedQueryCount = 0;
        for (Map.Entry<String, Float> entrySet : allAvPScores.entrySet()) {
            String key = entrySet.getKey();
            if (null != queryAPHash.get(key)) {

                // +++ NQC
//                Double[] qScore = queryScoreHash.get(key);
//                if (null != qScore) {
//                    double score[] = ArrayUtils.toPrimitive(qScore);
//
//                    nqc = returnNQC(score, noOfTopDocs);
//                    //System.out.println(query.qid+ " "+"\t" + returnNQC(score, noOfTopDocs));
//
//                }
//                else
//                    nqc = 0;

                // +++ AMB
                Float value = entrySet.getValue();
                if(value != 0)
                    value = 1/value;        // 

                allAvPScores.put(key, value);
                allAvP[processedQueryCount] = value;
                System.out.println(key + "\t" + value + "\t");
//                allAvP[processedQueryCount] = (alpha * value + (1-alpha) * nqc);
                allAP[processedQueryCount] = queryAPHash.get(key);
                processedQueryCount++;
            }
        }
        System.out.println("#############################");

        // AvNP
        processedQueryCount = 0;
        for (Map.Entry<String, Float> entrySet : allAvNPScores.entrySet()) {
            String key = entrySet.getKey();
            if (null != queryAPHash.get(key)) {

                // +++ NQC
//                Double[] qScore = queryScoreHash.get(key);
//                if (null != qScore) {
//                    double score[] = ArrayUtils.toPrimitive(qScore);
//
//                    nqc = returnNQC(score, noOfTopDocs);
//                    //System.out.println(query.qid+ " "+"\t" + returnNQC(score, noOfTopDocs));
//
//                }
//                else
//                    nqc = 0;

                // +++ AMB
                Float value = entrySet.getValue();
                if(value != 0)
                    value = 1/value;        // 

                allAvNPScores.put(key, value);
                allAvNP[processedQueryCount] = value;
                System.out.println(key + "\t" + value);
//                allAvNP[processedQueryCount] = (alpha * value + (1-alpha) * nqc);
                allAP[processedQueryCount] = queryAPHash.get(key);
                processedQueryCount++;
            }
        }
        PearsonsCorrelation cor = new PearsonsCorrelation();
        KendallsCorrelation tau = new KendallsCorrelation();
//        System.out.println("AvP - RHO, Tau: " + cor.correlation(allAP, allAvP) + " " +
//            tau.correlation(allAP, allAvP));
//        System.out.println("AvNP - RHO, Tau: " + cor.correlation(allAP, allAvNP) + " " +
//            tau.correlation(allAP, allAvNP));
    }

    public static void main(String[] args) throws Exception {

        AvP_AvNP obj;

        if(args.length != 0)
            obj = new AvP_AvNP(args[0]);
        else {
//            System.out.println("Usage: java NQC <properties.file>");
            args = new String[1];
            args[0] = "clueweb.properties";
//            args[0] = "qpp-trec678rb.properties";
//            args[0] = "qpp-wt10g.properties";
            obj = new AvP_AvNP(args[0]);
        }
        
//        for (obj.alpha = 0.0f; obj.alpha<=1.0; obj.alpha += 0.1) 
        obj.alpha = 1.0;
        {
            System.out.println("Alpha: " + obj.alpha + " ");
            obj.processAll();
        }
//        obj.alpha = 0.1;
//        {
//            System.out.println("Alpha: " + obj.alpha + " ");
//            obj.processAll();
//        }
    }

}
