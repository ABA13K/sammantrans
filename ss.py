
import requests

response = requests.get(
            url='https://proxy.scrapeops.io/v1/',
            params={
                'api_key': '89114a5a-195d-47a9-a67f-305304057cde',
                'url': 'https://www.amazon.com/Olay-Brightening-Vitamin-Moisturizing-Absorbing/dp/B0D7XWT4YS/?_encoding=UTF8&pd_rd_w=ZscyP&content-id=amzn1.sym.6cf7c564-ca00-466b-86d0-47197a966847&pf_rd_p=6cf7c564-ca00-466b-86d0-47197a966847&pf_rd_r=A4DANCYQAV2GA5HVY1J7&pd_rd_wg=7xlJa&pd_rd_r=5a5057c7-7b90-44a0-96b2-6174bdd92a70&ref_=pd_hp_d_btf_nta-top-sellers&th=1', 
      'render_js': 'true', 
            },
          )

print('Response Body: ', response.content)