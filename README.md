# MiniCV
A reusable Python library that emulates a subset of OpenCV's core functionality. Built entirely from scratch using only NumPy, Pandas, and Matplotlib, this library handles everything from low-level 2D convolutions and geometric transformations to advanced feature extraction and canvas drawing primitives.

# System Architecture 
<img width="208" height="150" alt="minicv_system_architecture" src="https://github.com/user-attachments/assets/d924b1e9-39f5-470b-b9d3-f0002169a758" />
<svg width="100%" viewBox="0 0 680 490" role="img" xmlns="http://www.w3.org/2000/svg">
<title style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">MiniCV library — system architecture</title>
<desc style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">Layered structural diagram showing MiniCV modules in three layers: foundation (io, core), processing (image_processing, transformations, feature_extractor), and canvas (drawing, text), all built on NumPy.</desc>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
<mask id="imagine-text-gaps-fhhd99" maskUnits="userSpaceOnUse"><rect x="0" y="0" width="680" height="490" fill="white"/><rect x="290.4892272949219" y="41.76536178588867" width="99.45523834228516" height="22.469280242919922" fill="black" rx="2"/><rect x="211.52911376953125" y="60.024627685546875" width="256.9417724609375" height="19.950742721557617" fill="black" rx="2"/><rect x="48" y="70.02462768554688" width="66.31990432739258" height="19.950742721557617" fill="black" rx="2"/><rect x="48" y="185.02462768554688" width="65.00816345214844" height="19.950742721557617" fill="black" rx="2"/><rect x="48" y="306.0246276855469" width="43.377593994140625" height="19.950742721557617" fill="black" rx="2"/><rect x="115.52803802490234" y="102.7653579711914" width="39.376609802246094" height="22.469280242919922" fill="black" rx="2"/><rect x="70.79119110107422" y="128.02462768554688" width="128.41761016845703" height="19.950742721557617" fill="black" rx="2"/><rect x="403.8215637207031" y="102.7653579711914" width="54.79258728027344" height="22.469280242919922" fill="black" rx="2"/><rect x="308.2023620605469" y="128.02462768554688" width="246.4615936279297" height="19.950742721557617" fill="black" rx="2"/><rect x="314" y="168.02462768554688" width="31.244009017944336" height="19.950742721557617" fill="black" rx="2"/><rect x="72.9277114868164" y="219.76536560058594" width="140.5782470703125" height="22.469280242919922" fill="black" rx="2"/><rect x="82.85757446289062" y="245.02462768554688" width="120.28483581542969" height="19.950742721557617" fill="black" rx="2"/><rect x="276.0010681152344" y="219.76536560058594" width="128.43544006347656" height="22.469280242919922" fill="black" rx="2"/><rect x="273.1152648925781" y="245.02462768554688" width="134.20199584960938" height="19.950742721557617" fill="black" rx="2"/><rect x="470.0234375" y="219.76536560058594" width="134.3872528076172" height="22.469280242919922" fill="black" rx="2"/><rect x="481.8946533203125" y="245.02462768554688" width="110.21067810058594" height="19.950742721557617" fill="black" rx="2"/><rect x="148.65521240234375" y="338.7653503417969" width="79.11927032470703" height="22.469280242919922" fill="black" rx="2"/><rect x="115.01566314697266" y="362.0246276855469" width="145.9686737060547" height="19.950742721557617" fill="black" rx="2"/><rect x="458.29730224609375" y="338.7653503417969" width="51.840614318847656" height="22.469280242919922" fill="black" rx="2"/><rect x="414.81317138671875" y="362.0246276855469" width="138.3737030029297" height="19.950742721557617" fill="black" rx="2"/><rect x="344" y="420.0246276855469" width="68.69414901733398" height="19.950742721557617" fill="black" rx="2"/><rect x="312.7035217285156" y="446.765380859375" width="55.031776428222656" height="22.469280242919922" fill="black" rx="2"/></mask></defs>

<rect x="30" y="28" width="620" height="390" rx="14" fill="var(--color-background-secondary)" stroke="var(--color-border-secondary)" stroke-width="0.5" style="fill:rgb(245, 244, 237);stroke:rgba(31, 30, 29, 0.3);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="340" y="53" text-anchor="middle" dominant-baseline="central" style="fill:rgb(20, 20, 19);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">MiniCV library</text>
<text x="340" y="70" text-anchor="middle" dominant-baseline="central" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">minicv/ package — all images as NumPy arrays</text>

<line x1="50" y1="184" x2="630" y2="184" stroke="var(--color-border-tertiary)" stroke-width="0.5" stroke-dasharray="4 4" mask="url(#imagine-text-gaps-fhhd99)" style="fill:rgb(0, 0, 0);stroke:rgba(31, 30, 29, 0.15);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-dasharray:4px, 4px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="50" y1="305" x2="630" y2="305" stroke="var(--color-border-tertiary)" stroke-width="0.5" stroke-dasharray="4 4" style="fill:rgb(0, 0, 0);stroke:rgba(31, 30, 29, 0.15);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-dasharray:4px, 4px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<text x="52" y="80" dominant-baseline="central" fill="var(--color-text-tertiary)" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">foundation</text>
<text x="52" y="195" dominant-baseline="central" fill="var(--color-text-tertiary)" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">processing</text>
<text x="52" y="316" dominant-baseline="central" fill="var(--color-text-tertiary)" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">canvas</text>

<g onclick="sendPrompt('Explain the io.py module in MiniCV in detail')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="50" y="88" width="170" height="74" rx="8" stroke-width="0.5" style="fill:rgb(238, 237, 254);stroke:rgb(83, 74, 183);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="135" y="114" text-anchor="middle" dominant-baseline="central" style="fill:rgb(60, 52, 137);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">io.py</text>
  <text x="135" y="138" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">I/O &amp; color conversion</text>
</g>

<g onclick="sendPrompt('Explain the core.py module in MiniCV in detail')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="232" y="88" width="398" height="74" rx="8" stroke-width="0.5" style="fill:rgb(238, 237, 254);stroke:rgb(83, 74, 183);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="431" y="114" text-anchor="middle" dominant-baseline="central" style="fill:rgb(60, 52, 137);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">core.py</text>
  <text x="431" y="138" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">normalize · clip · pad · convolve · spatial_filter</text>
</g>

<path d="M 310 162 L 310 186 L 143 186 L 143 203" fill="none" stroke="var(--color-border-secondary)" stroke-width="1" marker-end="url(#arrow)" style="fill:none;stroke:rgba(31, 30, 29, 0.3);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="318" y="178" dominant-baseline="central" fill="var(--color-text-tertiary)" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">uses</text>

<g onclick="sendPrompt('Explain the image_processing.py module in MiniCV in detail')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="50" y="203" width="185" height="78" rx="8" stroke-width="0.5" style="fill:rgb(225, 245, 238);stroke:rgb(15, 110, 86);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="143" y="231" text-anchor="middle" dominant-baseline="central" style="fill:rgb(8, 80, 65);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">image_processing.py</text>
  <text x="143" y="255" text-anchor="middle" dominant-baseline="central" style="fill:rgb(15, 110, 86);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">filters &amp; thresholding</text>
</g>

<g onclick="sendPrompt('Explain the transformations.py module in MiniCV in detail')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="247" y="203" width="185" height="78" rx="8" stroke-width="0.5" style="fill:rgb(225, 245, 238);stroke:rgb(15, 110, 86);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="340" y="231" text-anchor="middle" dominant-baseline="central" style="fill:rgb(8, 80, 65);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">transformations.py</text>
  <text x="340" y="255" text-anchor="middle" dominant-baseline="central" style="fill:rgb(15, 110, 86);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">resize · rotate · translate</text>
</g>

<g onclick="sendPrompt('Explain the feature_extractor.py module in MiniCV in detail')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="444" y="203" width="186" height="78" rx="8" stroke-width="0.5" style="fill:rgb(225, 245, 238);stroke:rgb(15, 110, 86);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="537" y="231" text-anchor="middle" dominant-baseline="central" style="fill:rgb(8, 80, 65);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">feature_extractor.py</text>
  <text x="537" y="255" text-anchor="middle" dominant-baseline="central" style="fill:rgb(15, 110, 86);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">hist · hog_lite · EHD</text>
</g>

<g onclick="sendPrompt('Explain the drawing.py module in MiniCV in detail')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="50" y="324" width="275" height="68" rx="8" stroke-width="0.5" style="fill:rgb(250, 236, 231);stroke:rgb(153, 60, 29);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="188" y="350" text-anchor="middle" dominant-baseline="central" style="fill:rgb(113, 43, 19);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">drawing.py</text>
  <text x="188" y="372" text-anchor="middle" dominant-baseline="central" style="fill:rgb(153, 60, 29);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">point · line · rect · polygon</text>
</g>

<g onclick="sendPrompt('Explain the text.py module in MiniCV in detail')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="337" y="324" width="293" height="68" rx="8" stroke-width="0.5" style="fill:rgb(250, 236, 231);stroke:rgb(153, 60, 29);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="484" y="350" text-anchor="middle" dominant-baseline="central" style="fill:rgb(113, 43, 19);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">text.py</text>
  <text x="484" y="372" text-anchor="middle" dominant-baseline="central" style="fill:rgb(153, 60, 29);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">text placement on arrays</text>
</g>

<line x1="340" y1="418" x2="340" y2="438" stroke="var(--color-border-secondary)" stroke-width="1" marker-end="url(#arrow)" style="fill:rgb(0, 0, 0);stroke:rgba(31, 30, 29, 0.3);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="348" y="430" dominant-baseline="central" fill="var(--color-text-tertiary)" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">all modules</text>

<g onclick="sendPrompt('How does MiniCV use NumPy internally across all modules?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="240" y="438" width="200" height="40" rx="8" stroke-width="0.5" style="fill:rgb(250, 238, 218);stroke:rgb(133, 79, 11);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="340" y="458" text-anchor="middle" dominant-baseline="central" style="fill:rgb(99, 56, 6);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">NumPy</text>
</g>
</svg>

# MiniCV Function Descriptions
1. Packaging & Project Structure
__init__.py
Initializes the MiniCV package and exposes public APIs for easy importing of core modules (I/O, filtering, transforms, features, drawing, utils).

2. Image I/O & Core Utilities
read_image(filepath)
Loads an image from disk into a NumPy array using Matplotlib backends. Supports common formats like PNG and JPG, returning the image as a multi-dimensional array for processing.

export_image(image, filepath)
Saves a NumPy array as an image file to disk. Handles both grayscale and RGB formats, ensuring proper conversion and file format compatibility (PNG/JPG).

rgb_to_gray(image)
Converts an RGB image to grayscale using weighted luminance coefficients (typically 0.299R + 0.587G + 0.114B), returning a 2D array.

gray_to_rgb(image)
Converts a grayscale image to RGB format by replicating the single channel across three channels, maintaining consistent shape conventions.

3. Core Operations (Foundation Functions)
normalize(image, mode)
Normalizes pixel values using one of at least three modes (e.g., min-max normalization to [0, 1], standardization, or scale to [0, 255]). Ensures consistent intensity ranges for processing.

clip_pixels(image, min_val, max_val)
Clips pixel values to a specified range [min_val, max_val], preventing overflow or underflow in image operations.

pad_image(image, pad_width, mode)
Adds padding around image borders using at least three modes (e.g., constant/zero padding, edge replication, or reflection). Essential for boundary handling in convolution operations.

convolve2d(image, kernel)
Performs 2D convolution on grayscale images by applying a kernel/filter. Validates kernel properties (odd dimensions, numeric type) and uses padding for boundary handling. Core function for all filtering operations.

filter2d(image, kernel)
Applies convolution-based filtering to both grayscale and RGB images. For RGB, processes each channel independently and recombines results.

4. Image Processing Techniques
mean_filter(image, kernel_size)
Applies a mean (box) filter for smoothing by averaging pixel values within a local neighborhood. Reduces noise while blurring edges.

gaussian_kernel(size, sigma)
Generates a Gaussian kernel of specified size using the given sigma (standard deviation). Returns a normalized 2D array representing the Gaussian distribution.

gaussian_filter(image, kernel_size, sigma)
Smooths an image using Gaussian filtering. Generates a Gaussian kernel and applies it via the convolution pipeline, providing edge-preserving noise reduction.

median_filter(image, kernel_size)
Applies median filtering for noise reduction, particularly effective against salt-and-pepper noise. Uses local neighborhood sorting (with justified loops) to compute median values.

threshold_global(image, threshold_value)
Binarizes an image using a global threshold value. Pixels above the threshold become white (255), others become black (0).

threshold_otsu(image)
Automatically determines optimal threshold using Otsu's method by maximizing inter-class variance. Returns both the threshold value and the binarized image.

threshold_adaptive(image, block_size, method, C)
Performs adaptive thresholding by calculating local thresholds for different regions. Supports mean and Gaussian methods with offset constant C for varying illumination conditions.

sobel_gradients(image)
Computes image gradients using Sobel operators in both x and y directions. Returns gradient magnitude and direction, useful for edge detection.

bitplane_slice(image, bit_plane)
Extracts a specific bit plane (0-7) from the image, isolating contribution of that bit position to pixel values. Useful for analyzing image compression and significance of bits.

compute_histogram(image)
Calculates the intensity histogram of a grayscale image, returning frequency distribution of pixel values (0-255).

histogram_equalization(image)
Enhances image contrast by redistributing pixel intensities to span the full range uniformly using cumulative distribution function (CDF) transformation.

laplacian_filter(image)
Applies Laplacian operator for edge detection and image sharpening. Detects regions of rapid intensity change by computing the second derivative of the image, highlighting edges and fine details.

gamma_correction(image, gamma)
Performs non-linear gamma correction to adjust image brightness and contrast. Values of gamma < 1 brighten the image, while gamma > 1 darkens it. Uses power-law transformation: output = input^gamma.

5. Geometric Transformations
resize(image, new_width, new_height, interpolation)
Resizes an image to specified dimensions using interpolation methods. Supports at least nearest-neighbor (required minimum) and bilinear interpolation for quality scaling.

rotate(image, angle, interpolation)
Rotates an image around its center by a specified angle (in degrees). Uses defined interpolation method to handle sub-pixel positioning and minimize artifacts.

translate(image, tx, ty)
Shifts an image by specified offsets (tx, ty) along x and y axes. Handles boundary regions appropriately.

6. Feature Extractors
color_histogram(image, bins)
Extracts a global color histogram descriptor by computing the distribution of color values across channels. For RGB images, computes per-channel histograms and concatenates them into a single feature vector characterizing overall color distribution.

basic_statistics(image)
Computes basic statistical descriptors for the entire image including mean, standard deviation, variance, min, max, and other moments. Returns a feature vector summarizing global intensity characteristics.

hog_lite(image, cell_size, bins)
Implements a lightweight Histogram of Oriented Gradients (HOG) descriptor. Divides the image into cells, computes gradient orientations, and builds histograms of gradient directions. Captures local shape and texture information through edge orientation patterns.

edge_histogram_descriptor(image, bins)
Extracts a gradient-based edge histogram descriptor by analyzing the distribution of edge orientations and magnitudes across the image. Computes edge information using gradient operators and creates a histogram representing edge patterns and their spatial distribution.

10. Drawing Primitives (Canvas Operations)
draw_point(image, x, y, color, thickness)
Draws a single point at coordinates (x, y) with specified color and thickness. Handles grayscale (scalar) and RGB (tuple) color formats.

draw_line(image, x1, y1, x2, y2, color, thickness)
Draws a line between two points using Bresenham's algorithm or equivalent. Supports color specification and thickness control with boundary clipping.

draw_rectangle(image, x, y, width, height, color, thickness, filled)
Draws a rectangle with top-left corner at (x, y). Supports both outline and filled modes with specified color and thickness.

draw_polygon(image, points, color, thickness, filled)
Draws a polygon defined by a list of vertices. Supports outline mode (required) and optionally filled mode, with proper edge rendering and boundary clipping.

12. Text Placement
put_text(image, text, x, y, font_scale, color, thickness)
Renders text string on the image at position (x, y). Supports font scaling for size control, color specification (grayscale/RGB), and thickness for text weight.
