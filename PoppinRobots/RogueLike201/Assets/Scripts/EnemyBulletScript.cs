using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemyBulletScript : MonoBehaviour
{
    public float speed;
    public float lifetime;
    public float distance;
    public LayerMask whatIsSolid;
    public int damage;
    private Transform player;
    private Vector2 target;
    private Vector2 bulletToPlayer;
    private Rigidbody2D rb2d;
    // Start is called before the first frame update
    void Start()
    {
        rb2d = this.GetComponent<Rigidbody2D>();
        player = GameObject.FindGameObjectWithTag("Player").transform;
        target = new Vector2(player.position.x, player.position.y);
        bulletToPlayer = player.position - transform.position;

        Invoke("DestroyProjectile", lifetime);
    }

    // Update is called once per frame
    void Update()
    {
        
        RaycastHit2D hitInfo = Physics2D.Raycast(transform.position, transform.up, distance, whatIsSolid);
        if (hitInfo.collider != null)
        {
            if (hitInfo.collider.CompareTag("Player"))
            {
                //Player take damage
                Debug.Log("Takin' damage");
                hitInfo.collider.GetComponent<PlayerScript>().takeDamage(damage);
                DestroyProjectile();
            }
        }

        rb2d.velocity = bulletToPlayer.normalized * speed * 1000 * Time.deltaTime;
    }

    void DestroyProjectile()
    {
        Destroy(this.gameObject);
    }
}
