const API = window.BACKEND_URL || "http://127.0.0.1:5000";
let file=null,volumeId=null,total=0,slice=0,uploading=false,analyzing=false;

const $=id=>document.getElementById(id);
const src=b=>b?(b.startsWith("data:")?b:"data:image/png;base64,"+b):"";

function message(t,type=""){ $("message").textContent=t; $("message").className="message "+type; }
function valid(f){return f&&f.name.toLowerCase().endsWith(".nrrd")&&f.name.toLowerCase().endsWith("_img_ct.nrrd");}

async function checkBackend(){
 try{
  const r=await fetch(API+"/health",{cache:"no-store"});
  if(!r.ok)throw Error();
  $("serverStatus").textContent="● AI BACKEND ONLINE";
  $("serverStatus").className="server online";
  message("Backend connected. Select your CT NRRD file.");
 }catch(e){
  $("serverStatus").textContent="● BACKEND OFFLINE";
  $("serverStatus").className="server offline";
  message("Backend not connected. Run: python backend\\app.py","error");
 }
}
checkBackend();

function selectFile(f){
 if(!valid(f)){file=null;$("uploadBtn").disabled=true;$("fileName").classList.add("hidden");message("Invalid file. Select a CT file ending with _IMG_CT.nrrd.","error");return;}
 file=f;$("fileName").textContent="✓ "+f.name;$("fileName").classList.remove("hidden");
 $("dropTitle").textContent="CT Scan Selected";$("dropText").textContent="Ready to upload";
 $("uploadBtn").disabled=false;message("File selected successfully.","success");
}
$("chooseBtn").onclick=e=>{e.stopPropagation();$("fileInput").click()};
$("dropZone").onclick=e=>{if(e.target.closest("button"))return;$("fileInput").click()};
$("fileInput").onchange=e=>selectFile(e.target.files[0]);
["dragenter","dragover"].forEach(x=>$("dropZone").addEventListener(x,e=>{e.preventDefault();$("dropZone").classList.add("drag")}));
["dragleave","drop"].forEach(x=>$("dropZone").addEventListener(x,e=>{e.preventDefault();$("dropZone").classList.remove("drag")}));
$("dropZone").addEventListener("drop",e=>selectFile(e.dataTransfer.files[0]));

$("uploadBtn").onclick=upload;
function upload(){
 if(!file||uploading)return;
 uploading=true;$("uploadBtn").disabled=true;$("uploadBox").classList.remove("hidden");$("progressBar").style.width="0%";message("Uploading CT volume...");
 const xhr=new XMLHttpRequest();
 xhr.open("POST",API+"/upload_nrrd",true);
 xhr.timeout=120000;
 xhr.upload.onprogress=e=>{if(e.lengthComputable){$("progressBar").style.width=(e.loaded/e.total*100)+"%";$("uploadText").textContent=`Uploading... ${Math.round(e.loaded/e.total*100)}%`}};
 xhr.onerror=()=>finishError("Cannot connect to backend. Keep backend/app.py running.");
 xhr.ontimeout=()=>finishError("Upload timed out. Check backend terminal.");
 xhr.onload=()=>{
  let d=null;try{d=JSON.parse(xhr.responseText)}catch(e){}
  if(xhr.status!==200||!d||!d.success){finishError((d&&d.error)||("Upload failed. HTTP "+xhr.status));return;}
  volumeId=d.volume_id;total=Number(d.total_slices);slice=Number(d.current_slice??Math.floor(total/2));
  $("viewer").classList.remove("hidden");
  $("volumeInfo").textContent=`${d.width} × ${d.height} × ${total} voxels`;
  $("sliceTotal").textContent=total;$("sliceNo").textContent=slice;$("sliceText").textContent=`Slice ${slice}`;
  $("rangeLast").textContent=total-1;$("slider").max=total-1;$("slider").value=slice;$("slider").disabled=false;
  $("prev").disabled=false;$("next").disabled=false;$("analyze").disabled=false;
  if(d.image_base64)$("ctImage").src=src(d.image_base64);else loadSlice();
  $("uploadText").textContent="Upload completed";message("✓ Upload successful — CT image loaded. Select a slice and click Analyze.","success");
  $("viewer").scrollIntoView({behavior:"smooth",block:"start"});
  uploading=false;
 };
 const form=new FormData();form.append("file",file);xhr.send(form);
}
function finishError(t){uploading=false;$("uploadBtn").disabled=false;message(t,"error");}

let sliceTimer;
$("slider").oninput=()=>{
 clearTimeout(sliceTimer);slice=Number($("slider").value);updateSliceUI();
 sliceTimer=setTimeout(loadSlice,180);
};
function updateSliceUI(){$("sliceNo").textContent=slice;$("sliceText").textContent=`Slice ${slice}`;}
async function loadSlice(){
 if(!volumeId||uploading)return;
 try{
  const d=await post("/get_slice",{volume_id:volumeId,slice_index:slice});
  $("ctImage").src=src(d.image_base64);
 }catch(e){message("Slice loading failed: "+e.message,"error")}
}
$("prev").onclick=()=>{if(slice>0){slice--;$("slider").value=slice;updateSliceUI();loadSlice()}};
$("next").onclick=()=>{if(slice<total-1){slice++;$("slider").value=slice;updateSliceUI();loadSlice()}};

$("analyze").onclick=analyze;
async function analyze(){
 if(!volumeId||analyzing)return;
 analyzing=true;$("analyze").disabled=true;$("analyze").textContent="⏳ Analyzing...";
 $("analysisStatus").textContent="Hybrid U-Net + Transformer is processing the selected CT slice...";
 message("AI segmentation running. Please wait...");
 try{
  const d=await post("/segment_nrrd",{volume_id:volumeId,slice_index:slice});
  $("results").classList.remove("hidden");
  $("outCT").src=src(d.ct_base64);$("outMask").src=src(d.mask_base64);$("outOverlay").src=src(d.overlay_base64);
  $("time").textContent=d.inference_seconds?Number(d.inference_seconds).toFixed(2)+" sec":"Complete";
  $("count").textContent=d.num_predicted_organs??0;
  const m=d.metrics||{};
  $("dice").textContent=num(m.dice);$("iou").textContent=num(m.iou);$("precision").textContent=num(m.precision);$("recall").textContent=num(m.recall);$("accuracy").textContent=num(m.accuracy);
  render(d.predicted_organs||[]);
  $("analysisStatus").textContent=`Analysis completed for slice ${slice}.`;
  message("✓ Segmentation completed successfully.","success");
  $("results").scrollIntoView({behavior:"smooth",block:"start"});
 }catch(e){$("analysisStatus").textContent="Analysis failed.";message("Analysis failed: "+e.message,"error")}
 finally{analyzing=false;$("analyze").disabled=false;$("analyze").textContent="✦ Analyze Current Slice";}
}
async function post(path,body){
 const r=await fetch(API+path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body),cache:"no-store"});
 let d;try{d=await r.json()}catch(e){throw Error("Backend returned invalid JSON")};
 if(!r.ok||!d.success)throw Error(d.error||"Request failed");
 return d;
}
function num(v){return typeof v==="number"&&isFinite(v)?v.toFixed(4):"N/A";}
function render(list){
 if(!list.length){
  $("organRows").innerHTML='<div class="empty">No OAR class predicted on this slice.</div>';
  return;
 }
 $("organRows").innerHTML=list.map(o=>{
  const center=(o.center_x!==undefined&&o.center_y!==undefined)
   ? `Center: (${Number(o.center_x).toFixed(0)}, ${Number(o.center_y).toFixed(0)})`
   : (o.location||"Center: —");
  const region=o.region||"—";
  const side=o.side||"—";
  return `<div class="organRow">
    <div class="organName">${esc(o.organ)}</div>
    <div><span class="tag">${esc(o.boundary||"Detected")}</span></div>
    <div class="locationCell">
      <strong>${esc(center)}</strong>
      <span>Region: ${esc(region)}</span>
      <span>Side: ${esc(side)}</span>
    </div>
    <div>${Number(o.pixel_area||0).toLocaleString()}</div>
  </div>`;
 }).join("");
}
function esc(v){return String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));}
