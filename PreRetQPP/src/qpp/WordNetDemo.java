/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package qpp;


public class WordNetDemo {
	
    public static void main(String args [])
    {
        WordNet w = new WordNet();
        w.searchWord("black bear");
        System.out.println(w.returnAllSensesCount("black bear"));
    }	
}
