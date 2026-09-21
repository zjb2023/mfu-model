function renderRecompute(){
 const prefix=`R${unit}:`,entries=Object.entries(G).filter(([k])=>k.startsWith(prefix));
 const pick=(r,kind)=>entries.find(([k,n])=>n.rank===r&&n.kind===kind)?.[0];
 const node=(key,x,y,title,type='reserved',w=145,boundary=false)=>{if(!key||!G[key])throw Error('Missing recompute anchor '+title);return {key,x,y,w,title,type,boundary}};
 const att=[];
 for(let r=8;r<16;r++){const y=62+(r-8)*100;
  att.push(node(pick(r,'recompute_local_compute'),80,y,'本地投影等计算','compute'),node(pick(r,'recompute_attention_compute'),900,y,'Attention前向计算','compute'));
 }
 for(let p=8;p<16;p+=2){const y=112+(p-8)*100;for(let j=0;j<4;j++)att.push(node(`${prefix}p${p}-${p+1}:CP${j}`,j<3?260+j*205:1110,y,`CP${j} · ${p}/${p+1}`,'comm'))}
 makeDiagram('recompute-attention',att,1300,870,prefix);
 const moe=[];
 for(let r=8;r<16;r++){const y=62+(r-8)*100;
  moe.push(node(pick(r,'recompute_expert_FC1'),340,y,'FC1 前向计算','compute'),node(pick(r,'recompute_expert_FC2'),550,y,'FC2 前向计算','compute'),node(pick(r,'recompute_EP_combine_visible'),990,y,'Combine可见处理','compute'),node(`r${r}:u${unit}:recompute`,1190,y,'重计算完成','reserved',145,true));
 }
 moe.push(node(`${prefix}EP8:dispatch_notify`,90,412,'EP8分发同步有效项','comm',175),node(`${prefix}EP8:combine_effective`,760,412,'EP8合并有效残余','comm',175));
 makeDiagram('recompute-moe',moe,1360,870,null);
 $('recompute-caption').textContent=`第${4-unit}层 · 重计算成本子图（前向行为，不是真反向）`;
}
