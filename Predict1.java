package kbPredict;


import java.lang.reflect.Type;
import java.util.Map;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

import com.google.gson.*;
import com.google.gson.reflect.TypeToken;


@Path("/1/{varX}")
public class Predict1 {
	
	private Predictor p= new Predictor();
	 @GET	 
	 @Produces(MediaType.APPLICATION_JSON)
	public String modelPredict(@PathParam("varX") String varX){

		 Map<String, String> map=p.predict(varX, "Fusion");
		 Gson gson = new Gson();
		//get the correct parameter for the generic type to ensure correct serialization
		 Type theType = new TypeToken<Map<String, String>>() {}.getType();
		 String res=gson.toJson(map, theType);
		 
		
		
        return res;
}
}
