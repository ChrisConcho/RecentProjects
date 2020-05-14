IF EXISTS DROP DATABASE PoppinRobots;
CREATE DATABASE PoppinRobots;
USE PoppinRobots;
CREATE TABLE Cosmetics (
	ID INT(2) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    GoldFit BOOLEAN NOT NULL,
    RedFit BOOLEAN NOT NULL,
    BlueFit BOOLEAN NOT NULL,
    BlackFit BOOLEAN NOT NULL
    );
    INSERT INTO Cosmetics ( GoldFit, RedFit,BlueFit,BlackFit)
		values(0,1,1,0),
         (0,0,0,0),
         (0,1,1,0);
CREATE TABLE Player(
	PlayerID INT(6) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    NAME VARCHAR(20) NOT NULL,
    HighScore INT(20) NOT NULL,
    NewGamePlus BOOLEAN NOT NULL,
    WeaponID INT(1) NOT NULL,
    Money INT(10) NOT NULL,
    FOREIGN KEY fk5(PlayerID) REFERENCES Cosmetics(ID)
    );
    INSERT INTO Player (NAME,HighScore, NewGamePlus,WeaponID,Money)
		values ('Chris', 1000,false, 0, 200),
        ('Bris', 100000,false, 0, 200),
        ('Eli', 900000,false, 0, 200);
