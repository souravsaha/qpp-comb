/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


public class InputOutput {

    /**
     * Reads the six column TREC formated .res file;
     * Returns a hash table, keyed by the query-id, with an array as field, 
     * containing the RSV of each of the documents retrieved for that query.
     * @param resPath
     * @return
     * @throws FileNotFoundException
     * @throws IOException 
     */
    public static HashMap<String, Double[]> readScoreFromResFile(String resPath) throws FileNotFoundException, IOException {

        HashMap<String, Double[]> resDocs = new HashMap<>();
        File fin = new File(resPath);
	FileInputStream fis = new FileInputStream(fin);
        //Construct BufferedReader from InputStreamReader
	BufferedReader br = new BufferedReader(new InputStreamReader(fis));

        String lastQid = "";
        String presentQid = "";
        List<Double> presentScores = new ArrayList<>();

 
	String line = null;
        do {
            line = br.readLine();
            if(line != null) {
    //            System.out.println(line);
                String tokens[] = line.split("\\s+");
                presentQid = tokens[0];
                if(lastQid.equals(presentQid)) {
                    presentScores.add(Double.parseDouble(tokens[4]));
                }
                else {
                    if(!lastQid.isEmpty()){
//                        System.out.println(lastQid);
                        resDocs.put(lastQid, presentScores.toArray(new Double[presentScores.size()]));
                        presentScores = new ArrayList<>();
                    }
                    presentScores.add(Double.parseDouble(tokens[4]));
                    lastQid = presentQid;
                }
            }
            else {
                if(presentScores != null) {
//                    System.out.println(lastQid);
                    resDocs.put(lastQid, presentScores.toArray(new Double[presentScores.size()]));
                }
            }
        } while (line != null);

	br.close();
        fis.close();

        return resDocs;
    }

    /**
     * Reads a two column file, with each query-id and corresponding AP;
     * Returns a hash table, keyed by the query-id with a AP value for that query
     * as field.
     * @param mapPath
     * @return
     * @throws FileNotFoundException
     * @throws IOException 
     */
    public static HashMap<String, Double> readAPFromFile(String mapPath) throws FileNotFoundException, IOException {

        HashMap<String, Double> queryMapHash = new HashMap<>();

        File fin = new File(mapPath);
	FileInputStream fis = new FileInputStream(fin);
	BufferedReader br = new BufferedReader(new InputStreamReader(fis));

        String line = null;
        String tokens[] = new String[2];
        do {
            line = br.readLine();
            if(line != null) {
                tokens = line.split(" ");
                queryMapHash.put(tokens[0], Double.parseDouble(tokens[1]));
            }
        } while (line != null);

        /*
        for(Map.Entry<String, Double> entrySet : queryMapHash.entrySet()) {
            System.out.println(entrySet.getKey() + " " + entrySet.getValue());
        }
        */

        return queryMapHash;
    }

    // for unit testing
    public static void main(String[] args) throws IOException {

//        InputOutput.readScoreFromResFile("trec6.baseline.res");
        InputOutput.readAPFromFile("trec6.baseline.map");
    }
}
