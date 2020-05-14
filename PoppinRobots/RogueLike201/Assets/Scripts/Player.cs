using System;
using System.Collections.Generic;
using UnityEngine;

[SerializableAttribute]
public class Player
{
    public int playerID { get; set; }
    public string username { get; set; }
    public int highScore { get; set; }
    public List<int> AllhighScore { get; set; }
    public bool newGamePlus { get; set; }
    public List<bool> cosmetic { get; set; }
    public int weaponID { get; set; }
    public int money { get; set; }
}
