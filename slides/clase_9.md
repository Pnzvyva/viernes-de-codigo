---
marp: true
theme: einstein
paginate: true
title: Viernes de codigo
description: Clase magistral — Redes Neuronales y LSTM aplicadas a la predicción de volatilidad
size: 16:9
math: mathjax
class: lead
---

 
# Viernes de código
## 🧠 Criptomonedas y mas allá
### _Sniper bots, triangulación de wallets, exchanges, market makers, Satoshi y Lamports._
**Juan Camilo Pinedo Campo** · Universidad del Norte  
_31 Octubre 2025_


---

# Estructura de la clase

1. ¿Qué es una criptomoneda y blockchain?
2. Wallets: claves, direcciones y privacidad
3. ¿Qué es un Satoshi? ¿Y un Lamport?
4. Exchanges: CEX vs DEX y su rol
5. Market makers: ¿quiénes son y qué hacen?
6. Sniper bots: concepto y efectos en el mercado
7. Triangulación de wallets y "whales"
8. Riesgos, ética y contramedidas
9. Conclusiones y recursos

---

# ¿Qué es una criptomoneda?

Es un activo digital nativo de una **blockchain** cuya transferencia y existencias es verficadas de manera descentralizada por protocolos informaticos.
- *Propiedades*  
  - *descentralización (en muchas)*  
  - *inmutabilidad*  
  - *pseudonimato*  
- *Ejemplos: Bitcoin (BTC), Ethereum (ETH), Solana (SOL).*

**Nota**  
No todos las criptomonedas son nativas  **algunas son tokens**.

---

# Entendamos la blockchain.

<div class="multicolumn vcenter"><div>

-  La podemos ver como un libro **publico** que registra transacciones que cualquiera puede ver. 

- Esas transacciones tienen formas especiales en que se registran los "protocolos" como consenso **PoW, PoS, variantes**

- Esos protocoles son transformados a bloques y cada bloque referencias al bloque anterior por lo que la hace inquebrantable **_Por ahora_**



</div><div>

![center w:350](https://i.pinimg.com/1200x/44/50/a5/4450a55190179cdec4d3da6c5ece356e.jpg)



</div></div>


---


# 🔐 ¿Y las wallets?

<div class="multicolumn vcenter"><div>

Una **Wallet** no es más que un contenedor con un par de claves: una _pública_ y una _privada_.

- **Clave privada:** permite firmar transacciones (control real).  
- **Dirección pública:** identificador que recibe fondos; derivada de la clave pública.  
- **Importante:** la custodia — tú controlas la clave o un tercero la custodia.  

</div><div>


</div><div>

<svg class="wallet-svg" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg" aria-label="Candado digital">
  <defs>
    <radialGradient id="halo" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#00b7ff" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="transparent" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="lockBody" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#0b62ff"/>
      <stop offset="100%" stop-color="#5ad2ff"/>
    </linearGradient>
  </defs>

  <!-- Halo -->
  <circle cx="150" cy="150" r="90" fill="url(#halo)">
    <animate attributeName="r" values="85;95;85" dur="5s" repeatCount="indefinite"/>
  </circle>

  <!-- Partículas -->
  <circle class="particle p1" cx="150" cy="40" r="3"/>
  <circle class="particle p2" cx="260" cy="150" r="3"/>
  <circle class="particle p3" cx="150" cy="260" r="3"/>
  <circle class="particle p4" cx="40" cy="150" r="3"/>

  <!-- Candado más realista -->
  <g transform="translate(100,100)">
    <path class="shackle" d="M50 15 a35 35 0 0 1 70 0 v30" />
    <rect class="lock" x="40" y="45" width="90" height="85" rx="10" ry="10"/>
    <circle class="keyhole" cx="85" cy="87" r="7"/>
  </g>
</svg>

<style>
.wallet-svg {
  width: 260px;
  height: auto;
}

/* animaciones */
@keyframes orbit { from { transform: rotate(0deg);} to { transform: rotate(360deg);} }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.7;} }
@keyframes swing { 0%,100%{transform:rotate(0);} 50%{transform:rotate(3deg);} }

/* partículas orbitando */
.particle {
  fill: #67e8f9;
  transform-origin: 150px 150px;
  animation: orbit 7s linear infinite;
  opacity: 0.8;
}
.p2 { animation-delay: 1.2s; animation-duration: 8s; }
.p3 { animation-delay: 2.4s; animation-duration: 9s; }
.p4 { animation-delay: 3.6s; animation-duration: 10s; }

/* candado */
.lock {
  fill: url(#lockBody);
  stroke: #b7e1ff;
  stroke-width: 2;
  animation: pulse 3s ease-in-out infinite;
  filter: drop-shadow(0 0 8px rgba(70,170,255,0.6));
}
.shackle {
  fill: none;
  stroke: #9dd1ff;
  stroke-width: 6;
  stroke-linecap: round;
  animation: swing 4s ease-in-out infinite;
}
.keyhole {
  fill: #fff;
  opacity: 0.8;
}
</style>

</div></div>

---




# ¿Satoshi, lamport?

- **Satoshi (sat)**: la unidad más pequeña de Bitcoin.  
  - 1 BTC = 100,000,000 satoshis.
  - Nombre en honor a Satoshi Nakamoto.

- **Lamport** (en Solana): lamports son la unidad mínima de SOL.  
  - Equivale a la fracción mínima definida por la cadena (similar a sat en Bitcoin).  

---

# 🏦 Exchanges: qué son y qué rol cumplen

| Tipo | Descripción | Ventajas | Riesgos |
|------|--------------|-----------|----------|
| **CEX**<br>(Centralized Exchange) | Intermediarios que **custodian fondos y ejecutan órdenes**.<br>Ej.: *Binance, Coinbase*. | - Alta **liquidez** y facilidad de uso.<br>- Servicios: **custodia, staking, trading fiat**. | - Custodia centralizada (**“no tus llaves, no tus fondos”**).<br>- **Regulación** y posibles bloqueos.<br>- Riesgo de **hackeos**. |
| **DEX**<br>(Decentralized Exchange) | Protocolos **on-chain** basados en **contratos inteligentes y pools de liquidez**.<br>Ej.: *Uniswap, Serum*. | - Usuario controla sus fondos (**custodia propia**).<br>- **Composabilidad** DeFi y transparencia. | - **Errores** en contratos.<br>- **Slippage** con poca liquidez.<br>- Dependencia de **oráculos**. |


<style>
table {
  width: 100%;
  font-size: 0.8rem;
  border-collapse: collapse;
}
th, td {
  padding: 0.4rem 0.5rem;
  border: 1px solid rgba(255,255,255,0.15);
  vertical-align: top;
}
th {
  background-color: rgba(11, 98, 255, 0.25);
  color: #fff;
  text-align: center;
}
tr:nth-child(even) {
  background-color: rgba(255,255,255,0.04);
}
</style>
}
</style>


---

# Market makers — “los fantasmas del poder crypto”

<div class="multicolumn vcenter"><div>

- Actores **(institucionales o algoritmos)** que proveen liquidez comprando y vendiendo constantemente.

- Beneficio: spread (diferencia entre precio compra/venta), rebates y servicios.

- Rol: reducen volatilidad, permiten ejecución eficiente; también pueden **influir en precios** si son muy grandes


</div><div>

![center w:350](https://i.pinimg.com/736x/07/e2/bd/07e2bd1a5e4ee05117b7c641f2d0f3be.jpg)



</div></div>




---

# Que son los Sniper bots

<div class="multicolumn vcenter"><div>

- **Qué son**: bots automatizados que intentan ejecutar órdenes en micro-ventanas de oportunidad (ej.: justo después de un listado, o front-running de liquidity events).

- **Tácticas** (conceptuales): monitorizar mempool / eventos on-chain, reaccionar en milisegundos.

- **Efectos**: pueden provocar precios de apertura exagerados, frontrunning y gas wars.


</div><div>

![center w:350](https://i.pinimg.com/736x/7e/6d/6b/7e6d6b7c0cbb05d59a290b8b9370353f.jpg)



</div></div>

---

# ¿Por qué existen?

<div class="multicolumn vcenter"><div>


- Velocidad: pequeños márgenes multiplicados a alta frecuencia.

- Oportunidades: listados de tokens, airdrops, discrepancias entre exchanges.

- Oportunidades: listados de tokens, airdrops, discrepancias entre exchanges.

</div><div>

![center w:350](https://i.pinimg.com/1200x/86/df/7d/86df7d9a315a220ffaffce0f1617fd26.jpg)


---

# Whale wallets y triangulación (concepto)

- **Whale** = wallet con grandes cantidades de un activo; sus órdenes mueven mercado.
- **Triangulación de wallets** (explicación conceptual):
  - Observadores pueden detectar desplazamiento de fondos entre wallets (ej.: exchange → wallet A → wallet B).
  - Triangulación puede referirse a: mover fondos a través de varias direcciones para dificultar rastreo o para crear apariencias de demanda/oferta.
- **Uso legítimo vs uso malicioso**:
  - Legítimo: gestión de tesorería, diversificación.
  - Malicioso: spoofing de mercado, lavado, manipulación.

---

# Cómo analizan los movimientos los investigadores (sin pasos técnicos)

- Observadores usan: exploradores, grafos de transacciones, etiquetas de exchanges, análisis de tiempo.

- Señales: movimientos sincronizados, saltos entre direcciones nuevas, interacción con mixers.

---

# Riesgos y consideraciones éticas
- Riesgos técnicos: bugs en smart contracts, fallas de 
custodia, ataques (51%, reentrancy).

- Riesgos de mercado: manipulación por whales, frontrunning por bots.

- Ética: ¿es justo aprovechar información pública para ejecutar bots que perjudican usuarios minoristas?

- Regulación: los exchanges y MM están cada vez más regulados; transparencia y controles AML/KYC.

---

#  Referencias y lecturas recomendadas

<div class="multicolumn vcenter"><div>

### 📘 Libros clave

- **Antonopoulos, A. (2017). _Mastering Bitcoin_**  
  <small>O’Reilly Media.</small>  
  ![w:130](https://images-na.ssl-images-amazon.com/images/I/81W9hZ8mBVL.jpg)

- **Narayanan, A. et al. (2016). _Bitcoin and Cryptocurrency Technologies_**  
  <small>Princeton University Press.</small>  
  ![w:130](https://images-na.ssl-images-amazon.com/images/I/71xW3I2h0tL.jpg)

- **Tapscott, D. & Tapscott, A. (2016). _Blockchain Revolution_**  
  <small>Penguin Random House.</small>  
  ![w:130](https://images-na.ssl-images-amazon.com/images/I/81AGh6x3sDL.jpg)

</div><div>

### 🌐 Artículos y documentación

- [📄 *Ethereum Whitepaper* – Vitalik Buterin (2014)](https://ethereum.org/en/whitepaper/)  
  ![w:100](https://ethereum.org/static/a110735d65f388b37390e12c49b88507/31987/eth-diamond-purple.webp)

- [📄 *Bitcoin: A Peer-to-Peer Electronic Cash System* – Satoshi Nakamoto (2008)](https://bitcoin.org/bitcoin.pdf)  
  ![w:100](https://bitcoin.org/img/icons/logo-bitcoin.svg?1681512486)


- [💡 *Solana Documentation: Lamports & Accounts*](https://docs.solana.com/developing/programming-model/accounts)  


</div></div>

<style>
img {
  border-radius: 6px;
  margin: 0.3rem 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
ul, li {
  line-height: 1.3;
  font-size: 0.9rem;
}
</style>

---

🐍 🐍 🐍 🐍
=

## Ahora vayamos a python.






