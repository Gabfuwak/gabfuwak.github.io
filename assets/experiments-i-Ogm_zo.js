import"./modulepreload-polyfill-B5Qt9EMX.js";/* empty css              */const o=[{id:"webgpu-triangle",title:"WebGPU Triangle",date:"January 28, 2026",description:"A hello world for WebGPU and for this website!",path:"/experiments/webgpu-triangle.html",tags:["Computer graphics"]}],c=["AI","Computer graphics","Robotics"];let l=null;function i(){const n=l?o.filter(t=>t.tags.includes(l)):o,e=document.getElementById("experiment-list");e.innerHTML=n.map(t=>`
    <li>
      <a href="${t.path}">${t.title}</a>
      ${t.tags.map(r=>`<span class="tag">${r}</span>`).join("")}
      <p>${t.date} - ${t.description}</p>
    </li>
  `).join(""),document.querySelectorAll("#tag-filters button").forEach(t=>{t.style.fontWeight=t.dataset.tag===(l??"All")?"bold":"normal"})}const s=document.getElementById("tag-filters"),a=document.createElement("button");a.textContent="All";a.dataset.tag="All";a.addEventListener("click",()=>{l=null,i()});s.appendChild(a);for(const n of c){const e=document.createElement("button");e.textContent=n,e.dataset.tag=n,e.addEventListener("click",()=>{l=n,i()}),s.appendChild(e)}i();
