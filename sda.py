from moviepy.editor import TextClip

# List available fonts
available_fonts = TextClip.list('font')
print(available_fonts)