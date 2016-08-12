package kbPredict;

import java.util.Map;
import java.util.TreeMap;
 
public class MapSort {
 
  public static Map sortByValue(Map<?, ?> unsortedMap){
    Map sortedMap = new TreeMap(new ValueComparator(unsortedMap));
    sortedMap.putAll(unsortedMap);
    return sortedMap;
  }
  public static Map sortByKey(Map unsortedMap){
    Map<Object, Object> sortedMap = new TreeMap<Object, Object>();
    sortedMap.putAll(unsortedMap);
    
    return sortedMap;
  }
}
