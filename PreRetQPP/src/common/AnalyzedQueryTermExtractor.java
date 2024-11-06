/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package common;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.search.Query;
import org.xml.sax.SAXException;


public class AnalyzedQueryTermExtractor {

    TRECQueryParser queryParser;

    public AnalyzedQueryTermExtractor(String queryFilePath) throws SAXException {
        queryParser = new TRECQueryParser(queryFilePath);
    }

    
    public static void main(String[] args) {

        if (args.length < 1) {
            args = new String[1];
            System.err.println("usage: java TRECQuery <input xml file>");
            args[0] = "trec123.xml";
        }

        String queryFilePath = "trec123.xml";
        try {

            EnglishAnalyzerWithSmartStopword obj;
            obj = new EnglishAnalyzerWithSmartStopword("smart-stopwords");
            Analyzer analyzer = obj.setAndGetEnglishAnalyzerWithSmartStopword();

            TRECQueryParser queryParser = new TRECQueryParser(queryFilePath, analyzer);
            queryParser.queryFileParse();

            for (TRECQuery query : queryParser.queries) {
                Query luceneQuery;
                luceneQuery = queryParser.getAnalyzedQuery(query);
                String[] queryTerms = luceneQuery.toString(queryParser.fieldToSearch).split(" ");
                for(int i=0; i<queryTerms.length; i++) 
                    System.out.println(queryTerms[i]);
                for(int i=0; i<queryTerms.length-1; i++) 
                    System.out.println(queryTerms[i] + " " + queryTerms[i+1] );
            }

        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
