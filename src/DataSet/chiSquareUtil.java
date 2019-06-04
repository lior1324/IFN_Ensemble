package DataSet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Hashtable;

public class chiSquareUtil {

    public Hashtable<Double, ArrayList<Double>> chiSquare=new Hashtable<>();

    public chiSquareUtil() {
        String COMMA_DELIMITER = ",";
        double[][] chi_Square = new double[1000][11];
        String pathToFile = "src\\DataSet\\chi_table.csv";
        boolean flag = true;
        try {
            try (BufferedReader br = new BufferedReader(new FileReader(pathToFile))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] values = line.split(COMMA_DELIMITER);
                    if (flag) {
                        for (int i = 0; i < values.length; i++) {
                            chiSquare.put(Double.parseDouble(values[i]), new ArrayList<>());
                        }
                        flag = false;
                    }
                    else{
                        chiSquare.get(0.001).add(Double.parseDouble(values[0]));
                        chiSquare.get(0.01).add(Double.parseDouble(values[1]));
                        chiSquare.get(0.05).add(Double.parseDouble(values[2]));
                        chiSquare.get(0.1).add(Double.parseDouble(values[3]));
                        chiSquare.get(0.015).add(Double.parseDouble(values[4]));
                        chiSquare.get(0.02).add(Double.parseDouble(values[5]));
                        chiSquare.get(0.025).add(Double.parseDouble(values[6]));
                        chiSquare.get(0.03).add(Double.parseDouble(values[7]));
                        chiSquare.get(0.035).add(Double.parseDouble(values[8]));
                        chiSquare.get(0.04).add(Double.parseDouble(values[9]));
                        chiSquare.get(0.045).add(Double.parseDouble(values[10]));
                    }
                }
            }
        } catch (Exception e) {

        }
    }

    public double chi_Stat(int df,double significance){
        if((df >= 0) && (df<=1000)){
            return this.chiSquare.get(significance).get(df);
        }
        return 0;
    }
}