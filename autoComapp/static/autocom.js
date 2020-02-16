$SCRIPT_ROOT = "{{url_for('editmd.editor')}}";

function autocompletemd(editor){
    //get un-complete text
    let cline=editor.getCursor().line;
    let cch=editor.getCursor().ch;
    editor.setSelection({line:cline,ch:cch},{line:cline,ch:0})
    let contenttocom=editor.getSelection();
    let contenttocoms=contenttocom.split(".");
    contenttocom=contenttocoms.pop();
    console.log("word need autocomplete: "+contenttocom);
    editor.setCursor({line:cline,ch:cch});

    if(contenttocom=='') return
    let contentChosed;//to save the chosed content num

    //get autocomplete list
    console.log("ask for json");
    $.ajax({
        type:'get',
        data:{'con':contenttocom},
        url:$SCRIPT_ROOT+'/getautocomp',
        success:function(data){
            console.log(data);
            let ali=data
            console.log(ali.length)
            console.log(ali[0].length)
    //get json here---------------------------------
    
    //alijson='[["edu",5,2,1,20],["ecnu",4,2,3,23],["wtenu",3,3,25,23]]';
    //console.log("got json: "+alijson);
    //ali=JSON.parse(alijson);

    //clear div
    autocompleteClose();

    //scroll
    function mousewheel(obj,fn){
        if (window.navigator.userAgent.indexOf('Firefox') != -1) {
            obj.addEventListener('DOMMouseScroll', wheelFn, true);
        } else obj.onmousewheel = wheelFn;
        function wheelFn(ev){
            ev=ev||event;
            var direct = ev.wheelDelta ? ev.wheelDelta < 0 : ev.detail > 0;
            fn && fn(direct);
            if (window.event) {
                ev.returnValue = false; 
                return false; 
            } else {
                ev.preventDefault();
            }
        };
    };
    //scroll

    //position
    let cursor=$(".CodeMirror-cursor");

    let atcbox=document.createElement("div");
    atcbox.setAttribute("id","autocombox");
    console.log(ali.length)
    let lb=10;
    if(ali.length<10) lb=ali.length; 
    lb=27*lb-1;
    let bb=$("body");
    bb.append(atcbox);
    $("#autocombox").css("height",lb+"px");
    $("#autocombox").css("left",cursor.offset().left+"px");
    let t=parseInt(cursor.css("height").slice(0,-2));
    $("#autocombox").css("top",cursor.offset().top+t+"px");
    let atcscroll=document.createElement("div");
    atcscroll.setAttribute("id","scroll");
    atcbox.append(atcscroll);
    if (ali.length<10){
        $("#scroll").css("visibility","hidden")
    }

    //make div
    let autocomlist=document.createElement("div");
    autocomlist.setAttribute("id","autocomlist");
    atcbox.append(autocomlist);
    lb=ali.length*27+1;
    $("#autocomlist").css("height",lb+"px");

    //scroll
    let dis_list=autocomlist.offsetHeight-atcbox.offsetHeight;
    let dis_span=atcbox.offsetHeight-atcscroll.offsetHeight;
    let wheel_rate=dis_span/dis_list;
    mousewheel(atcbox,function(dir){
        if(dir){
            let t=autocomlist.offsetTop-27;
            if(t<-dis_list) t=-dis_list;
            autocomlist.style.top=t+'px';
            atcscroll.style.top=-t*wheel_rate+'px';
        }else{
            let t=autocomlist.offsetTop+27;
            if(t>0) t=0;
            autocomlist.style.top=t+'px';
            atcscroll.style.top=-t*wheel_rate+'px';
        }
    });

    atcscroll.onmousedown=function(ev){
        ev=ev||window.event;
        let mt=ev.clientY-this.offsetTop;
        document.onmousemove=function(ev){
            ev = ev || window.event;
            let t = ev.clientY - mt;
            if (t <= 0) t = 0;
            if (t >= dis_span - 2) t = dis_span;
            move_rate = t / dis_span;
            autocomlist.style.top = -dis_list * move_rate + 'px';
            atcscroll.style.top = t + 'px';
        }
        document.onmouseup = function() {
            document.onmousemove = null;
        };
        return false;
    }


    //add data
    let atcFocus=0;
    
    for(let i=0;i<ali.length;i++){
        let atcitem=document.createElement("div");
        if(!i) atcitem.setAttribute("class","autocomitem-active")
        else atcitem.setAttribute("class","autocomitem")
        //click event listener
        atcitem.addEventListener("click",function(e){
            contentChosed=i;
            editor.insertValue(ali[i][0]);
            nchs=cch-contenttocom.length;
            nche=editor.getCursor().ch;
            tline=cline;
            tch=nche;
            window.removeEventListener("keydown",keyofautocom);
            stime=new Date();
            window.addEventListener("click",clickAfterAutocom);
            window.addEventListener("keydown",keyAfterAutocom);
            autocompleteClose();
        })
        let atcitemul=document.createElement("ul");
        for(let j=0;j<ali[i].length;j++){
            let atcitemli=document.createElement("li");
            if(!j){
                atcitemli.setAttribute("class","li-content");
                let t=ali[i][j];
                if(t.length>24){
                    t=t.slice(0,24);
                    t+="...";
                }
                atcitemli.innerHTML="<p>"+t+"</p>";
            }else if(j==3){
                atcitemli.setAttribute("class","li");
                atcitemli.innerHTML="<div class=\"atcms\" ><p>"+ali[i][j]+"ms"+"</p></div>";
            }else{
                atcitemli.setAttribute("class","li");
                atcitemli.innerHTML="<div class=\"atcper\" ><p>"+ali[i][j]+"%"+"</p></div>";
            }
            atcitemul.append(atcitemli);
        }
        atcitem.append(atcitemul);
        autocomlist.append(atcitem);
    }

    let nline,nchs,nche;//atc content postion
    nline=cline;
    let tline,tch;
    let diffper;

    let scrollflag;
    if(ali.length>10) scrollflag=1;
    let scrollm=0;

    let stime,etime;
    //keyboard event listener
    function keyofautocom(e){
        if(e.keyCode==40){//down
            atcFocus++;
            if(atcFocus>=ali.length) atcFocus=0;
            autocompleteSetHL(atcFocus);
            editor.setCursor({line:cline,ch:cch});
            if(scrollflag){
                scrollm++;
                if(atcFocus==0){
                    scrollm=0;
                    autocomlist.style.top='0px';
                        atcscroll.style.top='0px';
                }else{
                    if(scrollm>9){
                        scrollm=9;
                        let t=autocomlist.offsetTop-27;
                        if(t<-dis_list) t=-dis_list;
                        autocomlist.style.top=t+'px';
                        atcscroll.style.top=-t*wheel_rate+'px';
                    }
                }
            }
        }else if(e.keyCode==38){//up
            atcFocus--;
            if(atcFocus<0) atcFocus=ali.length-1;
            autocompleteSetHL(atcFocus);
            editor.setCursor({line:cline,ch:cch});
            if(scrollflag){
                scrollm--;
                if(atcFocus==ali.length-1){
                    scrollm=9;
                    autocomlist.style.top=-dis_list+'px';
                    atcscroll.style.top=dis_list*wheel_rate+'px';
                }else{
                    if(scrollm<0){
                        scrollm=0;
                        let t=autocomlist.offsetTop+27;
                        if(t>0) t=0;
                        autocomlist.style.top=t+'px';
                        atcscroll.style.top=-t*wheel_rate+'px';
                    }
                }
            }
        }else if(e.keyCode==13||e.keyCode==108){//enter
            contentChosed=atcFocus;
            editor.setSelection({line:cline,ch:cch},{line:cline+1,ch:0})
            editor.replaceSelection(ali[atcFocus][0]);
            nchs=cch-contenttocom.length;
            nche=editor.getCursor().ch;
            tline=cline;
            tch=nche;
            window.removeEventListener("keydown",keyofautocom);
            stime=new Date();
            window.addEventListener("click",clickAfterAutocom);
            window.addEventListener("keydown",keyAfterAutocom);
            autocompleteClose();
        }else if(e.keyCode==9){//tab
            contentChosed=atcFocus;
            let tabline=editor.getCursor().line;
            let tabch=editor.getCursor().ch;
            editor.setSelection({line:cline,ch:cch},{line:tabline,ch:tabch})
            editor.replaceSelection(ali[atcFocus][0]);
            cch-contenttocom.length;
            nche=editor.getCursor().ch;
            tline=cline;
            tch=nche;
            window.removeEventListener("keydown",keyofautocom);
            stime=new Date();
            window.addEventListener("click",clickAfterAutocom);
            window.addEventListener("keydown",keyAfterAutocom);
            autocompleteClose();
        }else if(e.keyCode==8||e.keyCode==37||e.keyCode==39||e.keyCode==27||e.keyCode==32||e.keyCode==35||e.keyCode==36||e.keyCode==46){//bs <- -> esc space end home delete
            window.removeEventListener("keydown",keyofautocom);
            autocompleteClose();
        }else if((e.keyCode>=96&&e.keyCode<=107)||(e.keyCode>=109&&e.keyCode<=111)){//numboard
            window.removeEventListener("keydown",keyofautocom);
            autocompleteClose();
        }else if((e.keyCode>=186&&e.keyCode<=192)||(e.keyCode>=219&&e.keyCode<=221)){//'=,-./`[]\'
            window.removeEventListener("keydown",keyofautocom);
            autocompleteClose();
        }
    }
    window.addEventListener("keydown",keyofautocom);
    
    //after-operation
    function clickAfterAutocom(e){
        tline=editor.getCursor().line;
        tch=editor.getCursor().ch;
        if(tline!=nline||(tline==nline&&(tch<nchs||tch>nche))){//click elsewhere
            endOP();
        }
    }

    function keyAfterAutocom(e){
        if(e.keyCode==13||e.keyCode==108){//enter
            endOP();
        }else if(e.keyCode==9){//tab
            endOP();
        }else if(e.keyCode>=33&&e.keyCode<=40){//<- up -> down end home
            tline=editor.getCursor().line;
            tch=editor.getCursor().ch;
            if(tline!=nline||(tline==nline&&(tch<nchs||tch>nche))){//move elsewhere
                endOP();
            }
        }else if(e.keyCode==32){//space
            if(tline==nline){
                if(tch==nchs){
                    tch++;
                    nche++;
                    nchs++;
                }else if(tch==nche){
                    tch++;
                    endOP();
                }else if(tch>nchs&&tch<nche){
                    tch++;
                    nche++;
                }
            }else endOP();
        }else if((e.keyCode==8)){//bs
            if(tline==nline){
                if(tch==nchs){
                    endOP();
                }else if(tch==nche){
                    tch--;
                    nche--;
                    if(nchs==nche) endOP();
                }else if(tch>nchs&&tch<nche){
                    tch--;
                    nche--;
                }
            }else endOP();
        }else if((e.keyCode==46)){//delete
            if(tline==nline){
                if(tch==nchs){
                    nche--;
                    if(nchs==nche) endOP();
                }else if(tch==nche){
                    endOP();
                }else if(tch>nchs&&tch<nche){
                    nche--;
                }
            }else endOP();
        }else if((e.keyCode>=65&&e.keyCode<=90)||(e.keyCode>=48&&e.keyCode<=87)){//abc 123
            tch++;
            nche++;
        }else if((e.keyCode>=96&&e.keyCode<=107)||(e.keyCode>=109&&e.keyCode<=111)){//numboard
            tch++;
            nche++;
        }else if((e.keyCode>=186&&e.keyCode<=192)||(e.keyCode>=219&&e.keyCode<=221)){//'=,-./`[]\'
            if(tline==nline){
                if(tch==nchs){
                    tch++;
                    nche++;
                    nchs++;
                }else if(tch==nche){
                    tch++;
                    endOP();
                }else if(tch>nchs&&tch<nche){
                    tch++;
                    nche++;
                }
            }else endOP();
        }
    }

    function endOP(){
        // console.log("tline,tch:"+tline+","+tch);
        // console.log("nline,nchs,nche:"+nline+","+nchs+","+nche+",");
        etime=new Date();
        calDiff();
        respondAuto(etime-stime);
        window.removeEventListener("keydown",keyAfterAutocom);
        window.removeEventListener("click",clickAfterAutocom);
    }


    function calDiff(){//calculate the difference between string chosed and edited
        let str1=ali[contentChosed][0];
        editor.setSelection({line:nline,ch:nchs},{line:nline,ch:nche});
        let str2=editor.getSelection();
        editor.setCursor({line:tline,ch:tch})
        //console.log("str1:"+str1+"str2:"+str2);
        if(str2.length<str1.length){
            let tstr=str1;
            str1=str2;
            str2=tstr;
        }
        if(str1==str2) diffper=0;
        else{
            let k=0;
            for(let i=0;i<str1.length;i++){
                for(let j=k;j<str2.length;j++){
                    if(str1[i]==str2[j]){
                        k++;
                        break;
                    }
                }
            }
            diffper=str2.length-k;
        }
    }

    function respondAuto(t){
        let res={
            'con':ali[contentChosed][0],
            'chosed':contentChosed,
            'time':t,
            'diff':diffper
        }
        //post json here-------------------------------
        console.log(res);
        $.ajax({
            type:'get',
            data:res,
            url:$SCRIPT_ROOT+'/postchosed',
            success:function(data){
                //console.log('chosed com post success!')
            },
            error:function(xhr,type,xxx){
                alert("error!");
            }
        })
    }

    //ajax1 end here
    },
    error:function(xhr,type,xxx){
        alert("error!");
    }
    })
    
}

function autocompleteClose(){
    let atclist=$("#autocombox");
    atclist.remove();
}
function autocompleteSetHL(atcFocus){
    let autocomlist=$("#autocomlist");
    let autocomitems=autocomlist.children();
    for(let i=0;i<autocomitems.length;i++) autocomitems[i].setAttribute("class","autocomitem");
    if(autocomitems.length>atcFocus) autocomitems[atcFocus].setAttribute("class","autocomitem-active");
}