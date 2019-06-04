package IFN;

import java.util.*;

public class IFNLayer {
    public int layerNum;
    public IFNLayer prevLayer;
    public Hashtable<String,Double> inputAttributesMI;
    public IFNLayer nextLayer;
    public ArrayList<IFNNode> NodesArr;
    public String splitBy;
    public InfoFuzzyNetwork network;
    public ArrayList<Double> splitBy_t_arr;
    public Hashtable<String,ArrayList<Double>> inputAttributesMI_t;
    public ArrayList<String> random_features_arr;

    public IFNLayer(int layerNum, IFNLayer prevLayer, InfoFuzzyNetwork network, ArrayList<String> random_features_arr){
        this.layerNum = layerNum;
        this.network = network;
        this.prevLayer = prevLayer;
        this.random_features_arr = random_features_arr;
        this.inputAttributesMI = new Hashtable<>();
        this.NodesArr = new ArrayList<>();
        this.splitBy = null;
        this.splitBy_t_arr = new ArrayList<>();
        this.inputAttributesMI_t = new Hashtable<>();
    }

    public void fit(){
        ArrayList<String> loop = this.NodesArr.get(0).ds.getColumnName();
        Hashtable<String,Hashtable<Double,Double>> temp_dic = new Hashtable<>();
        for(int i = 0; i < loop.size()-1; i++){
            String att = loop.get(i);
            if((!(this.network.splitted_att.contains(att))) && (this.random_features_arr.contains(att))){
                if(att.contains("nominal")){
                    double sum_mi = 0.0;
                    for(int j = 0; j < this.NodesArr.size(); j++){
                        sum_mi += this.NodesArr.get(j).input_att_mi.get(att);
                    }
                    this.inputAttributesMI.put(att,sum_mi);
                }
                else {
                    this.inputAttributesMI_t.put(att,new ArrayList<>());
                    ArrayList<Double> potential_t_in_network = this.network.ds.getColumnUniqueAtIndex(this.network.ds.getIndexByColumn(att));
                    Collections.sort(potential_t_in_network);
                    Hashtable<Double,Double> temp = new Hashtable<>();
                    for(int j = 0; j < potential_t_in_network.size(); j++){
                        temp.put(potential_t_in_network.get(j),fit_helper_continuous(att,potential_t_in_network.get(j)));
                    }
                    temp_dic.put(att,temp);
                    double key = get_key_max_val(temp_dic.get(att));
                    double val = temp_dic.get(att).get(key);
                    this.inputAttributesMI.put(att,val);
                    this.inputAttributesMI_t.get(att).add(key);
                    if(this.inputAttributesMI.get(att) > 0) recursize_descretization(key,att,false);

                }
            }
            else{
                this.inputAttributesMI.put(att,0.0);
            }
        }
        double max_mi = 0;
        ArrayList<String> keys = new ArrayList<>();
        keys.addAll(this.NodesArr.get(0).ds.getColumnName());
        keys.remove(this.NodesArr.get(0).ds.getColumnName().size()-1);
        for(int i = 0; i < keys.size(); i++){
            String key = keys.get(i);
            double val = this.inputAttributesMI.get(key);
            if((val > max_mi) && (!(this.network.splitted_att.contains(key)))){
                max_mi = val;
                this.splitBy = key;
                if(this.inputAttributesMI_t.keySet().contains(key)) this.splitBy_t_arr = this.inputAttributesMI_t.get(key);
                else this.splitBy_t_arr = null;
            }
        }
        if(max_mi == 0) this.splitBy = null;
        this.network.splitted_att.add(this.splitBy);
    }

    private double fit_helper_continuous(String att, double i){ //i is threshold here!!!!!!!!!!!!!!!!!!!!!
        double sum_mi = 0;
        for(int j = 0; j < this.NodesArr.size(); j++){
            IFNNode node = this.NodesArr.get(j);
            if(node.input_att_mi_t.get(att).containsKey(i)){
                sum_mi += node.input_att_mi_t.get(att).get(i);
            }
        }
        return sum_mi;
    }

    private void recursize_descretization(double threshold, String att, boolean flag){
        Hashtable<Double,Double> right_dic = new Hashtable<>();
        Hashtable<Double,Double> left_dic = new Hashtable<>();
        for(int i = 0; i < this.NodesArr.size(); i++){
            IFNNode node = this.NodesArr.get(i);
            node.global_discretization_in_node(threshold,att,this.inputAttributesMI_t.get(att),flag);
            if(!(node.right_interval.isEmpty())){
                Set<Double> keys = node.right_interval.keySet();
                Iterator it = keys.iterator();
                while(it.hasNext()) {
                    double key = (double) it.next();
                    double val = node.right_interval.get(key);
                    if(right_dic.containsKey(key)){
                        double old_val = right_dic.get(key);
                        right_dic.put(key,old_val+val);
                    }
                    else right_dic.put(key,val);
                }
            }
            if(!(node.left_interval.isEmpty())){
                Set<Double> keys = node.left_interval.keySet();
                Iterator it = keys.iterator();
                while(it.hasNext()) {
                    double key = (double) it.next();
                    double val = node.left_interval.get(key);
                    if(left_dic.containsKey(key)){
                        double old_val = left_dic.get(key);
                        left_dic.put(key,old_val+val);
                    }
                    else left_dic.put(key,val);
                }
            }
        }
        if(!(right_dic.isEmpty())){
            double key_right = get_key_max_val(right_dic);
            double val_right = right_dic.get(key_right);
            if(val_right > 0){
                this.inputAttributesMI_t.get(att).add(key_right);
                double old_val_right = this.inputAttributesMI.get(att);
                this.inputAttributesMI.put(att,old_val_right+val_right);
                recursize_descretization(key_right,att,false);
            }
        }
        if(!(left_dic.isEmpty())){
            double key_left = get_key_max_val(left_dic);
            double val_left = left_dic.get(key_left);
            if(val_left > 0){
                this.inputAttributesMI_t.get(att).add(key_left);
                double old_val_left = this.inputAttributesMI.get(att);
                this.inputAttributesMI.put(att,old_val_left+val_left);
                recursize_descretization(key_left,att,false);
            }
        }
    }

    private double get_key_max_val(Hashtable<Double,Double> ht){
        double max = -1;
        double max_key=0;
        Set<Double> keys = ht.keySet();
        ArrayList<Double> order_keys = new ArrayList<>(keys);
        Collections.sort(order_keys);
        for(int i = 0; i < order_keys.size(); i++){
            double threshold = order_keys.get(i);
            double mi_from_dic = ht.get(threshold);
            if(mi_from_dic > max){
                max = mi_from_dic;
                max_key = threshold;
            }
        }
        return max_key;
    }

    public IFNLayer buildNewLayer(ArrayList<String> att_random){
        if(this.splitBy == null){
            return null;
        }
        IFNLayer new_layer = new IFNLayer(this.layerNum+1,this,this.network,att_random);
        for(int i = 0; i < this.NodesArr.size(); i++){
            IFNNode node = this.NodesArr.get(i);
            String split_by = this.splitBy;
            if(split_by.contains("nominal")){
                if(node.input_att_mi.get(split_by) != 0.0){
                    node.split(split_by,null,new_layer.random_features_arr);
                    for(int j = 0; j < node.next_nodes.size(); j++){
                        new_layer.NodesArr.add(node.next_nodes.get(j));
                    }
                }
            }
            else{
                ArrayList<Double> t_arr_to_node = node.reduce_threshold(this.inputAttributesMI_t.get(split_by),split_by);
                if(t_arr_to_node.size() > 0){
                    node.split(split_by,t_arr_to_node,new_layer.random_features_arr);
                    for(int j = 0; j < node.next_nodes.size(); j++){
                        new_layer.NodesArr.add((node.next_nodes.get(j)));
                    }
                }
            }
        }
        new_layer.fit();
        return new_layer;
    }
}
