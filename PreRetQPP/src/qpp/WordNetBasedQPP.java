/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package qpp;

import common.InputOutput;
import common.TRECQuery;
import common.TRECQueryParser;
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.POS;
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
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.lucene.analysis.Analyzer;
import org.xml.sax.SAXException;
import static qpp.NQC.returnNQC;


public class WordNetBasedQPP {
    
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
    WordnetStemmer stemmer;
    HashMap<String, Integer> stopwordList;

    WordNetBasedQPP(String propPath) throws MalformedURLException, IOException, SAXException, Exception {

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

        String wnhome = "WordNet-3.0";
//        String wnhome = System.getenv("WNHOME");
        String path = wnhome + File.separator + "dict";
        url = new URL ("file", null, path) ;

        // construct the dictionary object and open it
        dict = new Dictionary ( url ) ;
        dict.open();
        stemmer = new WordnetStemmer(dict);

        // +++ making hashmap with smart-stopword list
        stopwordList = new HashMap<>();
        File file = new File("smart-stopwords");
        FileReader fileReader = new FileReader(file);
        BufferedReader bufferedReader = new BufferedReader(fileReader);
        String line;
        while ((line = bufferedReader.readLine()) != null) {
            stopwordList.put(line, 1);
        }
        fileReader.close();
        // ---
    }

    private int returnMaxOfAllSenses(List<String> stems) {

        int max = 0;
        int senses;

        IIndexWord idxWord;

        for (String stem : stems) {
            if(stopwordList.get(stem) == null) {
                idxWord = dict.getIndexWord(stem, POS.NOUN);
                if(null != idxWord) {
                    senses = idxWord.getWordIDs().size();
                    if (senses > max)
                        max = senses;
                }
/*
                idxWord = dict.getIndexWord(stem, POS.VERB);
                if(null != idxWord) {
                    senses = idxWord.getWordIDs().size();
                    if (senses > max)
                        max = senses;
                }

                idxWord = dict.getIndexWord(stem, POS.ADJECTIVE);
                if(null != idxWord) {
                    senses = idxWord.getWordIDs().size();
                    if (senses > max)
                        max = senses;
                }

                idxWord = dict.getIndexWord(stem, POS.ADVERB);
                if(null != idxWord) {
                    senses = idxWord.getWordIDs().size();
                    if (senses > max)
                        max = senses;
                }
*/
            }
        }

        return max;
    }

    private void processAll() throws Exception {

        double allAP[], allQPP[];
        int processedQueryCount = 0;
        allAP = new double [queries.size()];
        allQPP = new double [queries.size()];

        HashMap<String, Double[]> queryScoreHash = InputOutput.readScoreFromResFile(this.resPath);
        HashMap<String, Double> queryAPHash = InputOutput.readAPFromFile(this.mapPath);

        int sensesCount;
        int dictQuerytermCount;     // query terms which are present in the wordnet dict
        int OOVQuerytermCount;      // query terms which are NOT present in the wordnet dict
        float maxAmbScore;            // maximum no. senses associated with a Query

        HashMap<String, Float> allAMBScores = new HashMap<>();    // to contain all the scores of all the queries

        maxAmbScore = -1;

        for (TRECQuery query : queries) {

            float ambScore = 0;
            sensesCount = 0;
            dictQuerytermCount = 0;
            OOVQuerytermCount = 0;

            //System.out.println(query.qid + " " + query.qtitle + " ");
            String[] terms = query.qtitle.replaceAll("[^a-zA-Z0-9]", " ").split(" ");
            for (String term : terms) {
//                System.out.println(term);

                if(!term.isEmpty()){
                    //System.out.print(term + " : ");
                    List<String> stems = stemmer.findStems(term, null);
                    if(!stems.isEmpty())
                    {
                        term = stems.get(0);
                        int senses = returnMaxOfAllSenses(stems);
                        if( 0 != senses ) {
                            //System.out.println(term);
                            dictQuerytermCount ++; 
                            sensesCount += senses;
                        }
                        else {
                            //System.out.println("# - stopword");
                            //System.out.println("OOV1: " + term);
                            OOVQuerytermCount ++;                        
                        }
                    }
                    else {
                        //System.out.println("## - OOV");
                        //System.out.println("OOV2: " + term);
                        OOVQuerytermCount ++;
                    }
                }
            }

            if (dictQuerytermCount == 0)
                ambScore = 0;
            else
                ambScore = (float)sensesCount / (float)dictQuerytermCount;
            if(ambScore > maxAmbScore)
                maxAmbScore = ambScore;

            //System.out.println(sensesCount + " " + dictQuerytermCount + " " + OOVQuerytermCount + " " + ambScore);
            allAMBScores.put(query.qid, ambScore);
        }

        double nqc;
        double alpha;
        alpha = 1.0;
        for (alpha = 0.0f; alpha<=1.0; alpha += 0.1) {
            processedQueryCount = 0;
            System.out.print("Alpha: " + alpha + " ");
            for (Map.Entry<String, Float> entrySet : allAMBScores.entrySet()) {
                String key = entrySet.getKey();
                if (null != queryAPHash.get(key)) {

                    // +++ NQC
                    Double[] qScore = queryScoreHash.get(key);
                    if (null != qScore) {
                        double score[] = ArrayUtils.toPrimitive(qScore);

                        nqc = returnNQC(score, noOfTopDocs);
                        //System.out.println(query.qid+ " "+"\t" + returnNQC(score, noOfTopDocs));

                    }
                    else
                        nqc = 0;

                    // +++ AMB
                    Float value = entrySet.getValue();
                    if(value != 0)
                        value = 1/value;        // 

                    allAMBScores.put(key, value);
                    allQPP[processedQueryCount] = value;
                    allQPP[processedQueryCount] = (alpha * value + (1-alpha) * nqc);
                    allAP[processedQueryCount] = queryAPHash.get(key);
                    processedQueryCount++;
                }
            }

            PearsonsCorrelation cor = new PearsonsCorrelation();
            KendallsCorrelation tau = new KendallsCorrelation();
            System.out.println("RHO: " + cor.correlation(allAP, allQPP) + " " +
//                    " TAU: "  +
                    tau.correlation(allAP, allQPP));
        }
    }

    public static void main(String[] args) throws Exception {

        WordNetBasedQPP obj;

        if(args.length != 0)
            obj = new WordNetBasedQPP(args[0]);
        else {
//            System.out.println("Usage: java NQC <properties.file>");
            args = new String[1];
            args[0] = "qpp-bm25-trec2.properties";
//            args[0] = "qpp-trec678rb.properties";
//            args[0] = "qpp-wt10g.properties";
            obj = new WordNetBasedQPP(args[0]);
        }
        
        obj.processAll();
    }

}
