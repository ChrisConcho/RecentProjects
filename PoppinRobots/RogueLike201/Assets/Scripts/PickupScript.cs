using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PickupScript : MonoBehaviour
{
    public bool isWeapon;
    public int weaponId;
    public AudioSource audioData;

    // Start is called before the first frame update
    void Start()
    {
        audioData = GetComponent<AudioSource>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void OnTriggerStay2D(Collider2D collision)
    {
        bool interacted = false;
        if (collision.CompareTag("Player"))
        {
            if (isWeapon == false)
            {
                GameObject player = GameObject.FindGameObjectWithTag("Player");
                player.GetComponent<PlayerScript>().addCoin();

                SpriteRenderer spriteRend = GameObject.Find("HUD_Weapon").GetComponent<SpriteRenderer>();
                spriteRend.sprite = this.GetComponent<SpriteRenderer>().sprite;

                audioData.Play();
                interacted = true;
                StartCoroutine(WaitAudio());
            }
            else if (Input.GetKeyDown(KeyCode.E))
            {
                GameObject player = GameObject.FindGameObjectWithTag("Player");
                player.GetComponentInChildren<WeaponSwitchScript>().setWeapon(weaponId);
                interacted = true;
                Object.Destroy(gameObject);
            }
        }

        if (interacted == true)
        {
            SpriteRenderer sr = GetComponent<SpriteRenderer>();
            sr.enabled = false;
            Collider2D collider = GetComponent<Collider2D>();
            collider.enabled = false;
        }
    }
    IEnumerator WaitAudio()
    {
        yield return new WaitForSeconds(audioData.clip.length);
        Object.Destroy(gameObject);
    }
}


