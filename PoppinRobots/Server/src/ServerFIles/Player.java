package ServerFIles;

import java.util.List;
import java.util.Vector;

import com.google.gson.annotations.Expose;
import com.google.gson.annotations.SerializedName;

public class Player {

@SerializedName("PlayerID")
@Expose
private Integer playerID;
@SerializedName("username")
@Expose
private String username;
@SerializedName("HighScore")
@Expose
private Integer highScore;
@SerializedName("AllHighScore")
@Expose
private List<Integer> AllhighScore;
@SerializedName("NewGamePlus")
@Expose
private Boolean newGamePlus;
@SerializedName("Cosmetic")
@Expose
private List<Boolean> cosmetic = null;
@SerializedName("WeaponID")
@Expose
private Integer weaponID;
@SerializedName("Money")
@Expose
private Integer money;

public Integer getPlayerID() {
return playerID;
}

public void setPlayerID(Integer playerID) {
this.playerID = playerID;
}

public String getNAME() {
return username;
}

public void setNAME(String nAME) {
this.username = nAME;
}

public Integer getHighScore() {
return highScore;
}

public void setHighScore(Integer highScore) {
this.highScore = highScore;
}
public List<Integer> getAllHighScore(){
	return AllhighScore;
}
public void setAllHighScore(List<Integer> AllHighScore) {
	this.AllhighScore= AllHighScore;
}

public Boolean getNewGamePlus() {
return newGamePlus;
}

public void setNewGamePlus(Boolean newGamePlus) {
this.newGamePlus = newGamePlus;
}

public List<Boolean> getCosmetic() {
return cosmetic;
}

public void setCosmetic(List<Boolean> cosmetic) {
this.cosmetic = cosmetic;
}

public Integer getWeaponID() {
return weaponID;
}

public void setWeaponID(Integer weaponID) {
this.weaponID = weaponID;
}

public Integer getMoney() {
return money;
}

public void setMoney(Integer money) {
this.money = money;
}

}