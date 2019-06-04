package DataSet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Collections;
import java.util.*;

public class DataSet {

    private static String COMMA_DELIMITER =",";
    private ArrayList<String> columnName=null;
    private Hashtable<String,Integer> columnToIndex= new Hashtable<>();
    private Hashtable<Integer,String> indexToColumn= new Hashtable<>();
    private Hashtable<Integer,Integer> yToCountInstances=new Hashtable<>();
    public double[][] X = null;
    public int [] y=null;
    private Hashtable<String,HashSet<Double>> columnUnique = new Hashtable<>();
    public int n_samples;
    public int n_classes;

    public DataSet(String pathToFile, int inputAttributes, int n_classes)
    {
        this.n_classes = n_classes;
        int numOfSamples =0;
        try (BufferedReader br = new BufferedReader(new FileReader(pathToFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                numOfSamples++;
            }
        }
        catch(Exception e){

        }
        numOfSamples--;

        X = new double[numOfSamples][inputAttributes];
        y = new int [numOfSamples];
        this.n_samples = numOfSamples;
        int sampleCounter=0;
        boolean Flag=true; // first time.
        try {
            try (BufferedReader br = new BufferedReader(new FileReader(pathToFile))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] values = line.split(COMMA_DELIMITER);
                    if(!Flag){
                        //ArrayList<Double> row=new ArrayList<>();
                        for (int i = 0; i < values.length ; i++) {
                            double cell= Double.parseDouble(values[i]);
                            //Double cell = new Double(values[i]);
                            columnUnique.get(indexToColumn.get(i)).add(cell);
                            if(i!=inputAttributes){
                                X[sampleCounter][i]=cell;
                            }
                            else{
                                y[sampleCounter] = Integer.parseInt(values[i]);
                                /////////////
                                if(yToCountInstances.containsKey(Integer.parseInt(values[i]))){
                                    yToCountInstances.put(Integer.parseInt(values[i]),yToCountInstances.get(Integer.parseInt(values[i]))+1);
                                }
                                else{
                                    yToCountInstances.put(Integer.parseInt(values[i]),1);
                                }
                                ////////// added now .
                            }
                        }
                        sampleCounter++;
                    }
                    else
                    {
                        Flag=false;
                        columnName =new ArrayList<>(Arrays.asList(values));
                        for (int i = 0; i < values.length; i++) {
                            columnToIndex.put(values[i],i);
                            indexToColumn.put(i,values[i]);
                            columnUnique.put(values[i],new HashSet<>());
                        }
                    }
                }
            }
        }
        catch (Exception e){
            System.out.println("Cant rad the file got exception.");
            System.out.println(pathToFile);
            columnName = null;
            columnToIndex = null;
            indexToColumn = null;
            yToCountInstances = null;
            X = null;
            y = null;
            columnUnique = null;
            n_samples = 0;
            return;
        }
    }

    public DataSet(DataSet ds, ArrayList<Integer> indexes)
    {
        this.n_classes = ds.n_classes;
        this.columnName=ds.getColumnName();
        this.columnToIndex= ds.getColumnToIndex();
        this.indexToColumn= ds.getIndexToColumn();
        this.X = new double[indexes.size()][ds.columnName.size()-1];
        this.y= new int[indexes.size()];

        for (int i = 0; i < this.columnName.size(); i++) {
            this.columnUnique.put(this.indexToColumn.get(i), new HashSet<>());
        }
        for(int i = 0; i < indexes.size(); i++){
            for(int j = 0; j < this.columnName.size()-1; j++){
                this.X[i][j] = ds.X[indexes.get(i)][j];
                this.columnUnique.get(this.indexToColumn.get(j)).add(ds.X[indexes.get(i)][j]);
            }
            this.y[i] = ds.y[indexes.get(i)];
            this.columnUnique.get(this.indexToColumn.get(this.columnName.size()-1)).add((double)ds.y[indexes.get(i)]);
            if(yToCountInstances.containsKey(ds.y[indexes.get(i)])){
                yToCountInstances.put(ds.y[indexes.get(i)],yToCountInstances.get(ds.y[indexes.get(i)])+1);
            }
            else{
                yToCountInstances.put(ds.y[indexes.get(i)],1);
            }
        }
        this.n_samples = indexes.size();
    }

    public int getYCount(int index) {
        if(yToCountInstances.get(index) == null){
            return 0;
        }
        return yToCountInstances.get(index);
    }

    public ArrayList<Double> getColumnUniqueAtIndex(int index){
        String name = this.indexToColumn.get(index);
        return new ArrayList<>(this.columnUnique.get(name));
    }

    public String getColumnByIndex(int index){
        return this.indexToColumn.get(index);
    }

    public int getIndexByColumn(String column){
        return this.columnToIndex.get(column);
    }

    public ArrayList<String> getColumnName() {
        return this.columnName;
    }

    public double minAtColumn(String column){
        return Collections.min(this.columnUnique.get(column));
    }
    public double maxAtColumn(String column){
        return Collections.max(this.columnUnique.get(column));
    }


    public Hashtable<Integer, String> getIndexToColumn() {
        return indexToColumn;
    }

    public Hashtable<String, Integer> getColumnToIndex() {
        return columnToIndex;
    }

    public double[] getIlocX(int columnIdx){
        double[] res = new double[this.n_samples];
        for(int i = 0; i < this.n_samples; i++){
            res[i] = this.X[i][columnIdx];
        }
        return res;
    }

    public static double confusion_metrix(int[] preds, int[] y_true,int y_uniqe,boolean print){
        if(preds.length != y_true.length){
            System.out.println("preds and y true not the same size");
        }
        Hashtable<Integer,Hashtable<Integer,Integer>> res = new Hashtable<>();
        for(int i = 0; i < y_uniqe; i++){
            res.put(i,new Hashtable<>());
            res.get(i).put(1,0); //1st number is how much from this value in y true
            res.get(i).put(2,0); //2nd number is for how much classify correct for this value
            res.get(i).put(3,0); //3rd number is for how much classify uncorrect for this value
        }
        for(int i = 0; i < y_true.length; i++){
            int true_val = y_true[i];
            int pred_val = preds[i];
            res.get(true_val).put(1,res.get(true_val).get(1)+1);
            if(true_val == pred_val) res.get(true_val).put(2,res.get(true_val).get(2)+1);
            else res.get(true_val).put(3,res.get(true_val).get(3)+1);
        }
        double acc = 0;
        for(int i = 0; i < y_uniqe; i++){
            acc = acc + res.get(i).get(2);
        }
        acc = (double) acc / y_true.length;
        if(print){
            for(int i = 0; i < y_uniqe; i++){
                System.out.println("for y value ("+i+") was "+res.get(i).get(1)+" samples. "+res.get(i).get(2)+" classify correct and "+res.get(i).get(3)+" classify uncorrect.");
            }
            System.out.println("acc is "+acc+" and test error is "+(1-acc));
        }
        return acc;
    }
}
