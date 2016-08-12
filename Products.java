package kbPredict;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;


@Path("/products")
public class Products {
	
	 @GET	 
	 @Produces(MediaType.APPLICATION_JSON)
	public String productList(){
		 
		 return "{\"Fusion\": \"/predict/1/{search string}\",\n\"vCenter\": \"/predict/2/{search string}\",\n\"vSphere\": \"/predict/3/{search string}\",\n\"Horizon View\": \"/predict/4/{search string}\"}";

	 }
}
