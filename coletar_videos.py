"""Coleta opcional de exemplos públicos; respeite termos e limites dos sites."""
import asyncio
import csv
import os
import re
from pathlib import Path
from urllib.parse import quote
import aiohttp
from playwright.async_api import async_playwright

BASE = Path(__file__).resolve().parent
URL_ACESSIBILIDADE = 'http://www.acessibilidadebrasil.org.br/libras_3/'
URL_INES_MEDIA = 'https://dicionario.ines.gov.br/public/media/palavras/videos/'
PASTA_VIDEOS = BASE / 'videos_libras'
ARQUIVO_CSV = BASE / 'libras_completo.csv'
COLUNAS = ['letra','palavra','exemplo','video','url_video']

def palavras_ja_coletadas():
    if not ARQUIVO_CSV.exists(): return set()
    with ARQUIVO_CSV.open(encoding='utf-8-sig',newline='') as f:
        return {(r['letra'],r['palavra']) for r in csv.DictReader(f, delimiter=';')}

async def baixar_video_ines(session, nome_video):
    if not nome_video or '/' in nome_video or '\\' in nome_video or not re.fullmatch(r'[\w.-]+\.mp4', nome_video,re.I):
        return False, ''
    url = URL_INES_MEDIA + quote(nome_video)
    destino = PASTA_VIDEOS / nome_video
    if destino.exists() and destino.stat().st_size > 10240: return True,url
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=45)) as r:
            if r.status != 200: print('HTTP',r.status,url);return False,url
            conteudo = await r.read()
            if len(conteudo) <= 10240: return False,url
            temporario = destino.with_suffix('.part')
            temporario.write_bytes(conteudo)
            temporario.replace(destino)
            return True,url
    except (aiohttp.ClientError,asyncio.TimeoutError,OSError) as exc:
        print('Falha download:',exc)
        return False,url

async def raspar_dicionario_libras(limite_por_letra=None):
    PASTA_VIDEOS.mkdir(exist_ok=True)
    if not ARQUIVO_CSV.exists():
        with ARQUIVO_CSV.open('w',newline='',encoding='utf-8-sig') as f:
            csv.DictWriter(f,fieldnames=COLUNAS,delimiter=';').writeheader()
    concluidos=palavras_ja_coletadas()
    async with aiohttp.ClientSession() as session, async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(URL_ACESSIBILIDADE,wait_until='domcontentloaded',timeout=60000)
            for letra in map(chr,range(ord('A'),ord('Z')+1)):
                try:
                    link=page.locator(f"xpath=//a[normalize-space(text())='{letra}']")
                    if await link.count()==0: print('Sem letra',letra);continue
                    await link.first.click()
                    await page.wait_for_timeout(1200)
                    selects=page.locator('select')
                    if not await selects.count(): continue
                    select=selects.last
                    for j in range(await selects.count()):
                        candidate=selects.nth(j)
                        if 'SELECIONE' in (await candidate.inner_text()).upper():
                            select=candidate;break
                    palavras=list(dict.fromkeys(x.strip() for x in await select.locator('option').all_inner_texts()
                       if x.strip() and not x.strip().startswith('--') and x.strip()!=letra))
                    if limite_por_letra: palavras=palavras[:limite_por_letra]
                    for palavra in palavras:
                        if (letra,palavra) in concluidos:continue
                        try:
                            await select.select_option(label=palavra)
                            await select.dispatch_event('change')
                            await page.wait_for_timeout(650)
                            # Busca preferencialmente o video ativo, evitando capturar MP4 antigo no HTML.
                            srcs=await page.locator('video source, video').evaluate_all(
                               "els => els.map(e => e.currentSrc || e.src || e.getAttribute('src') || '').filter(Boolean)")
                            html=await page.content()
                            mp4s=[re.search(r'([\w-]+\.mp4)',s,re.I) for s in srcs]
                            found=next((m for m in mp4s if m),None)
                            if not found:
                                matches=re.findall(r'([\w-]+\.mp4)',html,re.I)
                                if len(set(matches))==1: found=re.search(r'([\w-]+\.mp4)',matches[0],re.I)
                            nome=found.group(1) if found else ''
                            ex=page.locator('#input-exemplo')
                            exemplo=''
                            if await ex.count():
                                tag=await ex.first.evaluate('e=>e.tagName.toLowerCase()')
                                exemplo=await ex.first.input_value() if tag in ('input','textarea') else await ex.first.inner_text()
                            exemplo=re.sub(r'\s+',' ',exemplo).strip()
                            ok,url=await baixar_video_ines(session,nome) if nome else (False,'')
                            if not ok:
                                print('Vídeo não encontrado/baixado:',letra,palavra)
                                continue  # Não marca como processado sem vídeo.
                            with ARQUIVO_CSV.open('a',newline='',encoding='utf-8-sig') as f:
                                csv.DictWriter(f,fieldnames=COLUNAS,delimiter=';').writerow(dict(
                                    letra=letra,palavra=palavra,exemplo=exemplo,video=nome,url_video=url))
                            concluidos.add((letra,palavra))
                            print(letra,palavra,nome)
                        except Exception as ex:
                            print('Erro em',palavra,ex)
                except Exception as ex:
                    print('Erro na letra',letra,ex)
        finally:
            await browser.close()
    print('Coleta concluída:',ARQUIVO_CSV)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--limite-por-letra',type=int,default=None)
    a=ap.parse_args()
    asyncio.run(raspar_dicionario_libras(a.limite_por_letra))
