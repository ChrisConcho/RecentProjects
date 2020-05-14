package ServerFIles;


import java.util.List;
import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

public class Example {

@SerializedName("Players")
@Expose
private List<Player> players = null;

public List<Player> getPlayers() {
return players;
}

public void setPlayers(List<Player> players) {
this.players = players;
}

}