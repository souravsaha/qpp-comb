/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package qpp;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.en.EnglishAnalyzer;



public class TrecDocAnalyzer {
    Properties prop;
    Analyzer analyzer;

    public TrecDocAnalyzer() {
        prop = null;
        analyzer = null;
    }

    public TrecDocAnalyzer(Properties prop) {
        this.prop = prop;
    }

    
    public final void setAnalyzer() {

        if(!prop.containsKey("stopFilePath")) {
            System.err.println("Error: Missing stopword file to used used by analyzer!\n");
            System.exit(1);
        }

        String stopFile = prop.getProperty("stopFilePath");
        List<String> stopwords = new ArrayList<>();

        String line;
        try {
            FileReader fr = new FileReader(stopFile);
            BufferedReader br = new BufferedReader(fr);
            while ( (line = br.readLine()) != null ) {
                stopwords.add(line.trim());
            }
            br.close(); fr.close();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }

        analyzer = new EnglishAnalyzer(StopFilter.makeStopSet(stopwords));
    }
    
    public Analyzer getAnalyzer() {

        if(analyzer == null) {
            System.err.println("Error: Analyzer is not set.\nCall setAnalyzer() before getAnalyzer()\n");
            System.exit(1);
        }
        return analyzer;
    }

}
