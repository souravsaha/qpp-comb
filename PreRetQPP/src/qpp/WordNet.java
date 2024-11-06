/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package qpp;


import java.net.URL;

import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.IWord;
import edu.mit.jwi.item.IWordID;
import edu.mit.jwi.item.POS;


class WordNet
{
    /*IDictionary is the main interface for acessing WordNet dictionary Files. 
     * Dictionary class implements IDictionary interface.
    */
    public IDictionary dictionary = null;

    WordNet() {
        try {
            String path = "/WordNet-3.0/dict";
            URL url = new URL("file", null, path);

            // construct the dictionary object and open it
            dictionary = new Dictionary(url);
            dictionary.open();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public int returnAllSensesCount(String key) {

        int totalSenses = 0;

        for ( POS pos : POS . values () ) {
            IIndexWord idxWord = dictionary.getIndexWord(key, pos);
            if(idxWord != null) {
                totalSenses += idxWord.getWordIDs().size();
            }
        }

        return totalSenses;
    }

    public int returnNounSenses(String key) {

        int totalSenses = 0;

        IIndexWord idxWord = dictionary.getIndexWord(key, POS.NOUN);
        if(idxWord != null) {
            totalSenses += idxWord.getWordIDs().size();
        }

        return totalSenses;
    }

    public void searchWord(String key) {

        /*  A word is having a different WordId in different synsets. Each Word is having a
         *  unique Index.  
        */

        //Get  the index associated with the word, 'book' with Parts of Speech NOUN.
        for ( POS pos : POS . values () ) {
            IIndexWord idxWord = dictionary.getIndexWord(key, pos);
            if(idxWord != null) {
                System.out.println("Word ->"+key);
                System.out.println("-------------");
                System.out.println("Number of senses in this POS: " + idxWord.getWordIDs().size());
                int i=1;

                /*getWordIDs() returns all the WordID associated with a index 
                 */		
                for(IWordID wordID : idxWord.getWordIDs()) {
                    //Construct an IWord object representing word associated with wordID  
                    IWord word = dictionary.getWord(wordID);
                    System.out.println("SENSE->"+i);
                    System.out.println("---------");

                    //Get the synset in which word is present.
                    ISynset wordSynset = word.getSynset();

                    System.out.print("Synset "+i+" {");

                    //Returns all the words present in the synset wordSynset
                    for(IWord synonym : wordSynset.getWords())
                    {
                        System.out.print(synonym.getLemma()+", ");
                    }
                    System.out.print("}"+"\n");

                    //Returns the gloss associated with the synset.
                    System.out.println("GLOSS -> "+wordSynset.getGloss());

                    System.out.println();
                    i++;
                }
            }
        }
    }
}