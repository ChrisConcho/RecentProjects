using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerScript : MonoBehaviour
{
    public Rigidbody2D rb;
    public Player playerData;
    public float moveVal;
    public GameObject crossHair;
    public GameObject weaponHolder;
    public float startTimeBtwShots;
    public int health;
    public bool alive;
    public int score;

    //private WeaponSwitchScript _switchScript;

    // Start is called before the first frame update
    void Start()
    {
        health = 100;
        score = 0;
        alive = true;
    }

    // Update is called once per frame
    void Update()
    {

        updateCrossHair();

        movePlayer();

        if (health <= 0)
        {
            alive = false;
            Destroy(this.gameObject);
        }
    }

    void Awake()
    {
        Debug.Log("Awake");
        DontDestroyOnLoad(gameObject);
    }

    public void addCoin()
    {
        playerData.money++;
    }

    void updateCrossHair()
    {
        Vector3 mousePosition = Input.mousePosition;
        mousePosition = Camera.main.ScreenToWorldPoint(mousePosition);
        Vector2 mouseDirection = new Vector2(mousePosition.x, mousePosition.y);
        //Debug.Log("mouseDirection: " + mouseDirection + ", player:" + rb.position);
        crossHair.transform.position = mouseDirection;

        Vector2 playertoCursor = new Vector2(mousePosition.x - rb.position.x, mousePosition.y - rb.position.y);
        float angle = Vector2.SignedAngle(new Vector2(0.0f, 1.0f), playertoCursor);
        rb.rotation = angle;
    }

    void movePlayer()
    {
        if (Input.GetKey(KeyCode.W))
        {
            //rb.rotation = 0.0f;
            rb.position = new Vector2(rb.position.x, rb.position.y + (moveVal * Time.deltaTime));
            //crossHair.transform.position = new Vector2(crossHair.transform.position.x, crossHair.transform.position.y +(moveVal*Time.deltaTime));
        }
        if (Input.GetKey(KeyCode.A))
        {
            //rb.rotation = 90.0f;
            rb.position = new Vector2(rb.position.x - (moveVal * Time.deltaTime), rb.position.y);
            //crossHair.transform.position = new Vector2(crossHair.transform.position.x - (moveVal * Time.deltaTime), crossHair.transform.position.y);
        }
        if (Input.GetKey(KeyCode.S))
        {
            //rb.rotation = 180.0f;
            rb.position = new Vector2(rb.position.x, rb.position.y - (moveVal * Time.deltaTime));
            //crossHair.transform.position = new Vector2(crossHair.transform.position.x, crossHair.transform.position.y - (moveVal * Time.deltaTime));
        }
        if (Input.GetKey(KeyCode.D))
        {
            //rb.rotation = 270.0f;
            rb.position = new Vector2(rb.position.x + (moveVal * Time.deltaTime), rb.position.y);
            //crossHair.transform.position = new Vector2(crossHair.transform.position.x + (moveVal * Time.deltaTime), crossHair.transform.position.y);
        }
    }

    public void takeDamage(int damage)
    {
        int newHealth = health - damage;

        if (newHealth < 0)
        {
            health = 0;
            this.alive = false;
        }
        else
        {
            health = newHealth;
        }
    }
}
